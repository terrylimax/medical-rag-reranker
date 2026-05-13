# Baseline Generation Examples (No Reranker)

- retriever: `hybrid`
- llm_model: `qwen/qwen3-4b-instruct-2507`
- top_k: `5`
- reranker_enabled: `False`
- num_examples: `300`

## Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Top docs**:

1. `medquad_ans_0006511-1` (score=0.0325) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the information is available, the ...
2. `medquad_ans_0006510-1` (score=0.0323) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the information is available, the ...
3. `medquad_ans_0006509-2` (score=0.0315) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is available, the tabl...
4. `medquad_ans_0001059-5` (score=0.0161) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndrome 1, X-linked - Ge...
5. `medquad_ans_0005115-2` (score=0.0156) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how often the symptom is ...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, and thrombocytopenia [medquad_ans_0006510-1].

**Citations detected**: `medquad_ans_0006510-1`
**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

## Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Top docs**:

1. `medquad_ans_0000312-1` (score=0.0164) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (muscle contractions), tw...
2. `medquad_ans_0000342-4` (score=0.0164) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000312-4` (score=0.0161) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the disorder and may neve...
4. `medquad_ans_0001013-4` (score=0.0161) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condition. Occasionally, a...
5. `medquad_ans_0000312-5` (score=0.0159) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Movement - uncontrolled...

**Answer**:

Yes, early-onset primary dystonia is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0000342-4]. Mutations in the TOR1A gene, which is associated with this condition, are inherited in this manner [medquad_ans_0000312-4]. In most cases, an affected person has one parent with the condition, and individuals who inherit the altered gene may or may not show signs or symptoms [medquad_ans_0001013-4].

**Citations detected**: `medquad_ans_0000342-4`, `medquad_ans_0000312-4`, `medquad_ans_0001013-4`
**Supported citations**: `medquad_ans_0000342-4`, `medquad_ans_0000312-4`, `medquad_ans_0001013-4`
**Unsupported citations**: _none_

## Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Top docs**:

1. `medquad_ans_0000689-1__dup2` (score=0.0306) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are easily treated and som...
2. `medquad_ans_0000265-5__dup1` (score=0.0272) - Loa loa parasites are found in West and Central Africa. Ten countries have areas where there are high rates of infection (i.e., where more than 40% of the people who live in that area report that they have had eye worm in the past). An e...
3. `medquad_ans_0000344-1__dup2` (score=0.0246) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons and Baylisascaris c...
4. `medquad_ans_0000660-1__dup1` (score=0.0240) - Baylisascaris roundworms are intestinal parasites found in many different animals. Baylisascaris infection in humans is uncommon but can be severe. While Baylisascaris can infect different types of animals, Baylisascaris procyonis, carri...
5. `medquad_ans_0000096-2__dup2` (score=0.0164) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns disease. The most commo...

**Answer**:

The provided context does not mention taeniasis or who is at risk for this specific parasitic infection. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000689-1__dup2, medquad_ans_0000265-5__dup1, medquad_ans_0000344-1__dup2]

**Citations detected**: `medquad_ans_0000689-1__dup2, medquad_ans_0000265-5__dup1, medquad_ans_0000344-1__dup2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000689-1__dup2, medquad_ans_0000265-5__dup1, medquad_ans_0000344-1__dup2`

## Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.0325) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is available, the table below ...
2. `medquad_ans_0000197-2__dup1` (score=0.0317) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is available, the table below ...
3. `medquad_ans_0000195-2__dup1` (score=0.0312) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is available, the table below ...
4. `medquad_ans_0000193-2__dup1` (score=0.0311) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is available, the table below ...
5. `medquad_ans_0000196-2__dup1` (score=0.0308) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is available, the table below ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy. These symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0000194-2__dup1].

**Citations detected**: `medquad_ans_0000194-2__dup1`
**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

## Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.0328) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0002946-1` (score=0.0247) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...
3. `medquad_ans_0000108-2__dup1` (score=0.0161) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
4. `medquad_ans_0005007-2` (score=0.0161) - What are the signs and symptoms of Potocki-Lupski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Potocki-Lupski syndrome. If the information is available, the table below includes how often t...
5. `medquad_ans_0005488-1` (score=0.0159) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how often the symptom is ...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Citations detected**: `medquad_ans_0003421-1`
**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

## Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Top docs**:

1. `medquad_ans_0002788-1` (score=0.0164) - Hemophilia B is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding or oozing following an injury or surgery. In severe cases of hemophilia, heavy bleeding occurs after minor...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0002786-1` (score=0.0161) - Hemophilia is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding following an injury, surgery, or having a tooth pulled. In severe cases, heavy bleeding occurs after minor t...
4. `medquad_ans_0000908-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000069-1__dup3` (score=0.0159) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed inside your body (interna...

**Answer**:

Yes, hemophilia is inherited. Hemophilia B is inherited in an X-linked recessive pattern and is caused by mutations in the F9 gene [medquad_ans_0002788-1]. Additionally, hemophilia is usually inherited, meaning the disorder is passed from parents to offspring [medquad_ans_0000069-1__dup3].

**Citations detected**: `medquad_ans_0002788-1`, `medquad_ans_0000069-1__dup3`
**Supported citations**: `medquad_ans_0002788-1`, `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

## Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.0328) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often the symptom is seen i...
2. `medquad_ans_0002946-1` (score=0.0308) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...
3. `medquad_ans_0006384-1` (score=0.0184) - What are the signs and symptoms of Waardenburg syndrome type 2B? The Human Phenotype Ontology provides the following list of signs and symptoms for Waardenburg syndrome type 2B. If the information is available, the table below includes h...
4. `medquad_ans_0000108-2__dup1` (score=0.0161) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
5. `medquad_ans_0006421-1` (score=0.0161) - What are the signs and symptoms of Wells-Jankovic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Wells-Jankovic syndrome. If the information is available, the table below includes how often t...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Citations detected**: `medquad_ans_0003284-1`
**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

## Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Top docs**:

1. `medquad_ans_0000775-1` (score=0.0323) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes)...
2. `medquad_ans_0000775-2` (score=0.0323) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
3. `medquad_ans_0000775-5` (score=0.0323) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglobinuria - MedlinePlu...
4. `medquad_ans_0000775-3` (score=0.0312) - Mutations in the PIGA gene cause paroxysmal nocturnal hemoglobinuria. The PIGA gene provides instructions for making a protein called phosphatidylinositol glycan class A. This protein takes part in a series of steps that produce a molecu...
5. `medquad_ans_0004747-1` (score=0.0303) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People with PNH have recurr...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells, affecting red blood cells (erythrocytes), white blood cells (leukocytes), and platelets (thrombocytes). It is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. This condition can cause hemolytic anemia, resulting in symptoms like fatigue, weakness, pallor, shortness of breath, and increased heart rate. Individuals are also at risk for infections due to white blood cell deficiency, abnormal blood clotting (thrombosis), especially in large abdominal veins, or severe bleeding (hemorrhage). Additionally, people with this disorder are

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Top docs**:

1. `medquad_ans_0000037_4-1` (score=0.0318) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. - Different factors ...
2. `medquad_ans_0000010-10` (score=0.0311) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large in...
3. `medquad_ans_0000010-11` (score=0.0308) - Colorectal cancer accounts for almost ten percent of all cancer deaths in the United States. The risk of developing colorectal cancer rises after age 50. It is common in both men and women.
4. `medquad_ans_0000010-1__dup5` (score=0.0305) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong -- cells become abno...
5. `medquad_ans_0000037_3-2` (score=0.0303) - Key Points - Avoiding risk factors and increasing protective factors may help prevent cancer. - The following risk factors increase the risk of colorectal cancer: - Age - Family history of colorectal cancer - Personal history - Inherited...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. The colon and rectum are parts of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large intestine, also called the large bowel. Cancer that begins in the colon is called colon cancer, and cancer that begins in the rectum is called rectal cancer. Cancer that begins in either of these organs may also be called colorectal cancer [medquad_ans_0000037_4-1, medquad_ans_0000010-10].

**Citations detected**: `medquad_ans_0000037_4-1, medquad_ans_0000010-10`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000037_4-1, medquad_ans_0000010-10`

## Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.0323) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0000544-4` (score=0.0311) - Both types of junctional epidermolysis bullosa are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carr...
3. `medquad_ans_0000544-1` (score=0.0308) - Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in response to minor i...
4. `medquad_ans_0003317-1` (score=0.0294) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Her...
5. `medquad_ans_0002121-1` (score=0.0293) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main types of epidermolysis...

**Answer**:

Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in response to minor injury or friction, such as rubbing or scratching. Researchers classify JEB into two main types: Herlitz JEB and non-Herlitz JEB. Although the types differ in severity, their features overlap significantly, and they can be caused by mutations in the same genes. Herlitz JEB is the more severe form, with blistering present from birth or early infancy over large regions of the body and affecting mucous membranes, leading to difficulties with eating and digestion, chronic malnutrition, scarring, granulation tissue formation, and potential respiratory complications. Infants with Herlitz JEB usually do not survive beyond the first year of life.

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.0325) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same signs and symptoms b...
2. `medquad_ans_0000668-4` (score=0.0318) - Molybdenum cofactor deficiency has an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. An affected individual usually inherits one altered copy of the gene from each parent. Par...
3. `medquad_ans_0000477-3` (score=0.0308) - Hereditary xanthinuria type I is caused by mutations in the XDH gene. This gene provides instructions for making an enzyme called xanthine dehydrogenase. This enzyme is involved in the normal breakdown of purines, which are building bloc...
4. `medquad_ans_0000416-3` (score=0.0277) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
5. `medquad_ans_0000668-5` (score=0.0161) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficiency, complementation...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C of the disorder [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis [medquad_ans_0000668-3]. In rare cases, an individual may inherit two mutated copies of the MOCS1 gene through uniparental isodisomy, a mechanism where the child receives two copies of the mutated gene from one parent due to an error in egg or sperm cell formation [medquad_ans_0000668-4]. The disorder follows an

**Citations detected**: `medquad_ans_0000668-3`, `medquad_ans_0000668-4`
**Supported citations**: `medquad_ans_0000668-3`, `medquad_ans_0000668-4`
**Unsupported citations**: _none_

## Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Top docs**:

1. `medquad_ans_0000756-1` (score=0.0325) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is e...
2. `medquad_ans_0000756-2` (score=0.0323) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
3. `medquad_ans_0000756-3` (score=0.0320) - Mutations in the SLC25A15 gene cause ornithine translocase deficiency. Ornithine translocase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver ce...
4. `medquad_ans_0004582-1` (score=0.0312) - What are the signs and symptoms of Ornithine translocase deficiency syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ornithine translocase deficiency syndrome. If the information is available, ...
5. `medquad_ans_0000755-1` (score=0.0299) - Ornithine transcarbamylase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system...

**Answer**:

Ornithine translocase deficiency is an inherited disorder caused by mutations in the SLC25A15 gene, which results in a defective mitochondrial ornithine transporter. This transporter is essential for moving ornithine across the inner membrane of mitochondria to the mitochondrial matrix, where it participates in the urea cycle. When the transporter is unstable or misshapen, ornithine cannot reach the matrix, disrupting the urea cycle. As a result, ammonia accumulates in the blood, which is toxic, especially to the nervous system. The disorder varies in severity and age of onset; infantile forms may present with lethargy, feeding difficulties, poor temperature regulation, seizures, or coma, while later-onset forms are usually less severe and may involve intolerance to high-protein foods, leading to episodes of vomiting, lethargy, ataxia, confusion, or blurred vision. Complications may include developmental delay, learning

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.0325) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance might be available. Cer...
2. `medquad_ans_0000045-14` (score=0.0303) - Some people think that Medicare and Medicaid are the same. Actually, they are two different programs. Medicaid is a state-run program that provides hospital and medical coverage for people with low income and little or no resources. Each...
3. `medquad_ans_0000045-28` (score=0.0285) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrollees in the fall. It i...
4. `medquad_ans_0000036-8` (score=0.0284) - Medicaid is a state health insurance program for those with low incomes and few assets. Each state runs its own program. The Federal Government requires that Medicaid programs cover a specific set of services; however, states can choose ...
5. `medquad_ans_0000045-9` (score=0.0256) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such as flu shots and dia...

**Answer**:

Yes, information about financial assistance is available. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2].

**Citations detected**: `medquad_ans_0000362-1__dup2`
**Supported citations**: `medquad_ans_0000362-1__dup2`
**Unsupported citations**: _none_

## Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Top docs**:

1. `medquad_ans_0004105-2` (score=0.0328) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often accompanied by long-ter...
2. `medquad_ans_0004105-1` (score=0.0323) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About half of affected indivi...
3. `medquad_ans_0000115-1__dup1` (score=0.0313) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fungus, and parasites. ...
4. `medquad_ans_0000598-1__dup2` (score=0.0312) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enters the body through t...
5. `medquad_ans_0000115-2__dup1` (score=0.0299) - Anyone experiencing symptoms of meningitis or encephalitis should see a doctor immediately. Antibiotics for most types of meningitis can greatly reduce the risk of dying from the disease. Antiviral medications may be prescribed for viral...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are the same as those found in other types of meningitis, including high fever, severe headache, nausea, vomiting, stiff neck, photophobia

**Citations detected**: `medquad_ans_0004105-1`
**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

## Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0003319-1` (score=0.0323) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. Unlike other types of...
2. `medquad_ans_0003319-3` (score=0.0311) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality cannot be determine...
3. `medquad_ans_0003331-3` (score=0.0310) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...
4. `medquad_ans_0003331-1` (score=0.0308) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
5. `medquad_ans_0003319-2` (score=0.0301) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex, muscle spasm (clonu...

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant or autosomal recessive fashion [medquad_ans_0003319-3].

**Citations detected**: `medquad_ans_0003319-3`
**Supported citations**: `medquad_ans_0003319-3`
**Unsupported citations**: _none_

## Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Top docs**:

1. `medquad_ans_0000219-4__dup3` (score=0.0277) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men age 60 to 69 - 30 perc...
2. `medquad_ans_0000219-8` (score=0.0263) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give up if the first treat...
3. `medquad_ans_0000219-11` (score=0.0260) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long enough for sex...
4. `medquad_ans_0000219-5__dup2` (score=0.0258) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain your relationship. Se...
5. `medquad_ans_0000218-5__dup1` (score=0.0247) - Talking about bladder control problems is not easy for some people. You may feel embarrassed to tell your doctor. But talking about the problem is the first step in finding an answer. Also, you can be sure your doctor has heard it all be...

**Answer**:

Erectile dysfunction (ED) affects men of all races and ages globally, with prevalence increasing with age: about 12 percent of men younger than 60, 22 percent of men aged 60 to 69, and 30 percent of men age 70 or older [medquad_ans_0000219-4__dup3].

Treatment options include addressing the underlying cause of ED. This may involve lifestyle changes such as quitting smoking, reducing alcohol intake, increasing physical activity, and avoiding illegal drug use [medquad_ans_0000219-8]. Your doctor may also adjust medications you are taking for other health conditions, as some medications can contribute to ED [medquad_ans_0000219-8]. Counseling can help couples manage the emotional impact of ED and strengthen their relationship [medquad_ans_0000219

**Citations detected**: `medquad_ans_0000219-4__dup3`, `medquad_ans_0000219-8`
**Supported citations**: `medquad_ans_0000219-4__dup3`, `medquad_ans_0000219-8`
**Unsupported citations**: _none_

## Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.0328) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often becomes necessa...
2. `medquad_ans_0000202-5` (score=0.0313) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Registry: Collagen Type VI...
3. `medquad_ans_0000576-5` (score=0.0304) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystrophy - Genetic Testi...
4. `medquad_ans_0000395-5` (score=0.0291) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscular Dystrophy - Geneti...
5. `medquad_ans_0001026-5` (score=0.0259) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrophy These resources f...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Citations detected**: `medquad_ans_0006267-3`
**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

## Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Top docs**:

1. `medquad_ans_0004005-3` (score=0.0164) - What causes microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1)? Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) has been shown to be caused by mutations in the RNU4ATAC gene.
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0004005-5` (score=0.0161) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected individuals, with most ...
4. `medquad_ans_0000043-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0004006-1` (score=0.0159) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and symptoms of MOPD2 may incl...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type II (MOPD2) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Citations detected**: `medquad_ans_0004006-1`
**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

## Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000742-3` (score=0.0164) - Ochoa syndrome can be caused by mutations in the HPSE2 gene. This gene provides instructions for making a protein called heparanase 2. The function of this protein is not well understood. Mutations in the HPSE2 gene that cause Ochoa synd...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000742-1` (score=0.0161) - Ochoa syndrome is a disorder characterized by urinary problems and unusual facial expressions. The urinary problems associated with Ochoa syndrome typically become apparent in early childhood or adolescence. People with this disorder may...
4. `medquad_ans_0000565-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000742-2` (score=0.0159) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.

**Answer**:

Yes, Ochoa syndrome is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, but they do not show signs and symptoms of the condition [medquad_ans_0000785-4].

**Citations detected**: `medquad_ans_0000785-4`
**Supported citations**: `medquad_ans_0000785-4`
**Unsupported citations**: _none_

## Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.0315) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management of various health c...
2. `medquad_ans_0003266-1` (score=0.0313) - Isodicentric chromosome 15 syndrome is a chromosome abnormality that affects many different parts of the body. As the name suggests, people with this condition have an extra chromosome (called an isodicentric chromosome 15) which is made...
3. `medquad_ans_0000531-2` (score=0.0291) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
4. `medquad_ans_0000531-3` (score=0.0267) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic material and has two ...
5. `medquad_ans_0006556-1` (score=0.0260) - 15q13.3 microduplication syndrome is a rare chromosome abnormality first described in 2009. Since only a small number of individuals with this microduplication have been reported, the full range of effects is still being discovered. What...

**Answer**:

Treatment for isodicentric chromosome 15 syndrome is based on the signs and symptoms present in each person and may include diagnostic tests, drug therapy, genetic counseling, surgery and rehabilitation, and palliative care [medquad_ans_0000531-5].

**Citations detected**: `medquad_ans_0000531-5`
**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

## Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Top docs**:

1. `medquad_ans_0000998-1` (score=0.0318) - Troyer syndrome is part of a group of genetic disorders known as hereditary spastic paraplegias. These disorders are characterized by progressive muscle stiffness (spasticity) and the development of paralysis of the lower limbs (parapleg...
2. `medquad_ans_0006224-3` (score=0.0313) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of the nervous system. ...
3. `medquad_ans_0000998-2` (score=0.0308) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.
4. `medquad_ans_0000272-4__dup1` (score=0.0303) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cause Troyer syndrome a...
5. `medquad_ans_0006224-2` (score=0.0301) - What are the signs and symptoms of Troyer syndrome? The signs and symptoms of Troyer syndrome can vary, and some people are more severely affected than others. Symptoms typically begin in early childhood. Most affected children have dela...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene, which gives the body instructions to make a protein called spartin. It is thought that the syndrome is caused by a loss of function of the spartin protein, although the exact role of this protein in cellular functions is not fully understood [medquad_ans_0006224-3].

**Citations detected**: `medquad_ans_0006224-3`
**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

## Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Top docs**:

1. `medquad_ans_0003672-1` (score=0.0325) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Timothy syndrome also ...
2. `medquad_ans_0000983-3` (score=0.0320) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1.2, is one of several...
3. `medquad_ans_0000983-1` (score=0.0280) - Timothy syndrome is a rare disorder that affects many parts of the body including the heart, digits (fingers and toes), and the nervous system. Timothy syndrome is characterized by a heart condition called long QT syndrome, which causes ...
4. `medquad_ans_0000983-4` (score=0.0255) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene, and o...
5. `medquad_ans_0000341-3` (score=0.0215) - A small percentage of all cases of familial atrial fibrillation are associated with changes in the KCNE2, KCNJ2, and KCNQ1 genes. These genes provide instructions for making proteins that act as channels across the cell membrane. These c...

**Answer**:

The genetic changes related to Timothy syndrome are mutations in the CACNA1C gene. These mutations change the structure of the CaV1.2 calcium channels, causing them to stay open much longer than usual, which leads to an abnormal influx of calcium ions into cells. This results in disrupted cellular functions, such as altered heart rhythms and potential developmental issues [medquad_ans_0003672-1][medquad_ans_0000983-3]. All reported cases of Timothy syndrome are due to changes in the CACNA1C gene [medquad_ans_0000983-1].

**Citations detected**: `medquad_ans_0003672-1`, `medquad_ans_0000983-3`, `medquad_ans_0000983-1`
**Supported citations**: `medquad_ans_0003672-1`, `medquad_ans_0000983-3`, `medquad_ans_0000983-1`
**Unsupported citations**: _none_

## Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000875-1` (score=0.0301) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hands and feet are invol...
2. `medquad_ans_0000061-11` (score=0.0282) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your doctor will look for ...
3. `medquad_ans_0000061-3__dup3` (score=0.0279) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by inflammation of the j...
4. `medquad_ans_0004326-1` (score=0.0263) - What are the signs and symptoms of Negative rheumatoid factor polyarthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Negative rheumatoid factor polyarthritis. If the information is available, th...
5. `medquad_ans_0005651-2` (score=0.0257) - What are the signs and symptoms of Sjogren syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sjogren syndrome. If the information is available, the table below includes how often the symptom is ...

**Answer**:

The symptoms of rheumatoid arthritis include pain, swelling, and stiffness of the joints, with small joints in the hands and feet being most commonly affected [medquad_ans_0000875-1]. The joint pain and stiffness are often worse in the morning or after a long rest [medquad_ans_0000875-1]. The disease typically affects joints in a symmetrical pattern, such as both hands or both knees [medquad_ans_0000875-1][medquad_ans_0000061-3__dup3]. In addition to joint symptoms, rheumatoid arthritis can cause inflammation of other tissues and organs, including the eyes, lungs, and blood vessels [medquad_ans_0000875-1]. Other signs and symptoms may include a loss of energy, low fever, weight loss, and anemia [medquad

**Citations detected**: `medquad_ans_0000875-1`, `medquad_ans_0000061-3__dup3`
**Supported citations**: `medquad_ans_0000875-1`, `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

## Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.0323) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, ...
2. `medquad_ans_0002796-4` (score=0.0318) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti-inflammatory drugs ...
3. `medquad_ans_0002796-2` (score=0.0315) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table below includes how often...
4. `medquad_ans_0002796-3` (score=0.0313) - Can Henoch-Schonlein purpura be inherited? The cause of Henoch-Schonlein purpura is currently unknown. Some evidence suggests that genetic predisposition may contribute to the development of this disease in some cases. Only a few familie...
5. `medquad_ans_0000149-1__dup3` (score=0.0296) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, digestive tract, and ...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, it may develop as an immune response to an infection. HSP is usually seen in children, but it may affect people of any age. Most cases go away on their own without treatment. [medquad_ans_0002796-1]

**Citations detected**: `medquad_ans_0002796-1`
**Supported citations**: `medquad_ans_0002796-1`
**Unsupported citations**: _none_

## Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.0323) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information is available, the ta...
2. `medquad_ans_0003163-1` (score=0.0318) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information is available, the ta...
3. `medquad_ans_0003167-1` (score=0.0288) - What are the signs and symptoms of Immunodeficiency without anhidrotic ectodermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency without anhidrotic ectodermal dysplasia. If ...
4. `medquad_ans_0006509-2` (score=0.0288) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is available, the tabl...
5. `medquad_ans_0006510-1` (score=0.0280) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the information is available, the ...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Citations detected**: `medquad_ans_0003166-1`
**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

## Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.0328) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger-like projections th...
2. `medquad_ans_0005069-3` (score=0.0297) - What genes are related to Hutchinson-Gilford progeria syndrome? Mutations in the LMNA gene cause Hutchinson-Gilford progeria syndrome. The LMNA gene provides instructions for making a protein called lamin A. This protein plays an importa...
3. `medquad_ans_0001051-3` (score=0.0226) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells and tissues throughou...
4. `medquad_ans_0000896-1` (score=0.0161) - Senior-Lken syndrome is a rare disorder characterized by the combination of two specific features: a kidney condition called nephronophthisis and an eye condition known as Leber congenital amaurosis. Nephronophthisis causes fluid-filled ...
5. `medquad_ans_0000606-3` (score=0.0161) - Mutations in the OCRL gene cause Lowe syndrome. The OCRL gene provides instructions for making an enzyme that helps modify fat (lipid) molecules called membrane phospholipids. By controlling the levels of specific membrane phospholipids,...

**Answer**:

Mutations in one of at least five genes are associated with Senior-Lken syndrome. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia, which are involved in signaling pathways that transmit information between cells. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling pathways within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. Some individuals with Senior-Lken syndrome do not have identified mutations in these five genes, and in such cases, the genetic cause is unknown [medquad_ans_0000896-3].

**Citations detected**: `medquad_ans_0000896-3`
**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

## Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.0283) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed as having lymphedema ...
2. `medquad_ans_0000610-1` (score=0.0164) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and immune cells through...
3. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0000610-5` (score=0.0161) - These resources address the diagnosis or management of lymphedema-distichiasis syndrome: - Gene Review: Gene Review: Lymphedema-Distichiasis Syndrome - Genetic Testing Registry: Distichiasis-lymphedema syndrome - MedlinePlus Encyclopedia...
5. `medquad_ans_0000989-2` (score=0.0161) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

Lymphedema-distichiasis syndrome affects an estimated 1 in 50,000 people [medquad_ans_0000989-2]. Additionally, it is reported to affect about 1 per million people worldwide [medquad_ans_0000359-2]. The prevalence may vary by region, with higher rates in certain areas of Quebec, Canada [medquad_ans_0000359-2].

**Citations detected**: `medquad_ans_0000989-2`, `medquad_ans_0000359-2`
**Supported citations**: `medquad_ans_0000989-2`, `medquad_ans_0000359-2`
**Unsupported citations**: _none_

## Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Top docs**:

1. `medquad_ans_0000013-26` (score=0.0269) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when they find out their ...
2. `medquad_ans_0005761-3` (score=0.0164) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation. Prenatal...
3. `medquad_ans_0000013-25` (score=0.0164) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for treatment or monitor...
4. `medquad_ans_0001483-5` (score=0.0161) - Is genetic testing available for congenital bilateral absence of the vas deferens (CBAVD)? GeneTests lists the names of laboratories that are performing genetic testing for CBAVD. To view the contact information for the clinical laborato...
5. `medquad_ans_0000397-1__dup1` (score=0.0161) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourself or a family membe...

**Answer**:

Yes, there is information about genetic testing. Genetic testing can cost anywhere from less than $100 to more than $2,000, and health insurance companies may cover part or all of the cost. Testing cannot tell you everything about inherited diseases; for example, a positive result does not always mean you will develop a disease, and it is hard to predict how severe symptoms may be. Geneticists and genetic counselors can explain what a particular test will or will not tell you and help decide whether to undergo testing. Some people may feel emotional reactions such as guilt, anger, anxiety, or depression upon learning results. Additionally, concerns about genetic discrimination exist, but the Genetic Information Nondiscrimination Act (GINA) of 2008 protects individuals from discrimination by health insurance providers or employers, though it does not apply to long-term care, disability, or life insurance. [medquad_ans_0000013-

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.0320) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxygen-rich blood from ...
2. `medquad_ans_0000654-1` (score=0.0318) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary malformations). In peo...
3. `medquad_ans_0000654-3` (score=0.0294) - Microcephaly-capillary malformation syndrome results from mutations in the STAMBP gene. This gene provides instructions for making a protein called STAM binding protein. This protein plays a role in sorting damaged or unneeded proteins s...
4. `medquad_ans_0000149-5` (score=0.0293) - These resources address the diagnosis or management of CM-AVM: - Gene Review: Gene Review: RASA1-Related Disorders - Genetic Testing Registry: Capillary malformation-arteriovenous malformation These resources from MedlinePlus offer infor...
5. `medquad_ans_0000641-1` (score=0.0288) - Megalencephaly-capillary malformation syndrome (MCAP) is a disorder characterized by overgrowth of several tissues in the body. Its primary features are a large brain (megalencephaly) and abnormalities of small blood vessels in the skin ...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries affecting blood circulation. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications of AVM

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Top docs**:

1. `medquad_ans_0002000-1` (score=0.0301) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes how often the sympt...
2. `medquad_ans_0001992-1` (score=0.0299) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the symptom is seen in pe...
3. `medquad_ans_0001999-1` (score=0.0299) - What are the signs and symptoms of Dystonia 6, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 6, torsion. If the information is available, the table below includes how often the sympt...
4. `medquad_ans_0001995-1` (score=0.0299) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the symptom is seen in pe...
5. `medquad_ans_0000141-2__dup1` (score=0.0296) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table below includes how oft...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive course, and retrocollis [medquad_ans_0001992-1].

**Citations detected**: `medquad_ans_0001992-1`
**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

## Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.0325) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable. Vision, phy...
2. `medquad_ans_0000899-5` (score=0.0311) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia: Hypopituitarism Th...
3. `medquad_ans_0005548-3` (score=0.0161) - What causes septo-optic dysplasia? In most cases of septo-optic dysplasia, the cause of the disorder is unknown. Researchers suspect that a combination of genetic and environmental factors may play a role in causing this disorder. Propos...
4. `medquad_ans_0005548-1` (score=0.0159) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the midline of the brain...
5. `medquad_ans_0000745-5` (score=0.0159) - These resources address the diagnosis or management of oculodentodigital dysplasia: - Genetic Testing Registry: Oculodentodigital dysplasia - MedlinePlus Encyclopedia: Webbing of the fingers or toes - UC Davis Children's Hospital: Cleft ...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Citations detected**: `medquad_ans_0005548-5`
**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

## Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Top docs**:

1. `medquad_ans_0000208-14` (score=0.0328) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The cause of benign pro...
2. `medquad_ans_0000208-7` (score=0.0318) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stone...
3. `medquad_ans_0000208-11` (score=0.0318) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men who are prescribed m...
4. `medquad_ans_0000208-1__dup4` (score=0.0310) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic obstruction. The pro...
5. `medquad_ans_0000208-10` (score=0.0299) - Treatment options for benign prostatic hyperplasia may include - lifestyle changes - medications - minimally invasive procedures - surgery A health care provider treats benign prostatic hyperplasia based on the severity of symptoms, how ...

**Answer**:

The complications of benign prostatic hyperplasia may include acute urinary retention, chronic or long-lasting urinary retention, blood in the urine, urinary tract infections (UTIs), bladder damage, kidney damage, and bladder stones [medquad_ans_0000208-14]. Most men with benign prostatic hyperplasia do not develop these complications, however, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

**Citations detected**: `medquad_ans_0000208-14`, `medquad_ans_0000208-7`
**Supported citations**: `medquad_ans_0000208-14`, `medquad_ans_0000208-7`
**Unsupported citations**: _none_

## Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Top docs**:

1. `medquad_ans_0000001-3__dup1` (score=0.0284) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-threatening.
2. `medquad_ans_0000150-2__dup1` (score=0.0164) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
3. `medquad_ans_0000001-1__dup1` (score=0.0161) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers that connect the cere...
4. `medquad_ans_0000019-2__dup1` (score=0.0161) - There is no cure or standard treatment for anencephaly. Treatment is supportive.
5. `medquad_ans_0000096-1__dup2` (score=0.0159) - Septo-optic dysplasia (SOD) is a rare disorder characterized by abnormal development of the optic disk, pituitary deficiencies, and often agenesis (absence) of the septum pellucidum (the part of the brain that separates the anterior horn...

**Answer**:

There is no specific treatment for the absence of the septum pellucidum by itself, as it is not life-threatening and does not have a standard course of treatment. However, if the absence is part of a broader condition such as septo-optic dysplasia, treatment is symptomatic and supportive, addressing associated symptoms like hormonal deficiencies, seizures, vision problems, and developmental delays [medquad_ans_0000001-3\_\_dup1][medquad_ans_0000001-1__dup1].

**Citations detected**: `medquad_ans_0000001-3__dup1`, `medquad_ans_0000001-1__dup1`
**Supported citations**: `medquad_ans_0000001-3__dup1`, `medquad_ans_0000001-1__dup1`
**Unsupported citations**: _none_

## Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.0325) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (i.e., PLEVA). In FUM...
2. `medquad_ans_0000114-2__dup1` (score=0.0306) - What are the signs and symptoms of Acute febrile neutrophilic dermatosis? The most obvious signs of acute febrile neutrophilic dermatosis are distinctive skin lesions that usually develop according to a specific pattern. Typically, a ser...
3. `medquad_ans_0004907-2` (score=0.0294) - What are the signs and symptoms of pityriasis lichenoides et varioliformis acuta? Pityriasis lichenoides et varioliformis acuta (PLEVA) is the acute form of a skin condition called pityriasis lichenoides. It is characterized by the sudde...
4. `medquad_ans_0002322-3` (score=0.0164) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD and Epstein-Barr vir...
5. `medquad_ans_0002322-1` (score=0.0159) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sores, then form a red-...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include initial red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (PLEVA). These lesions can progress rapidly to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become life threatening [

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Top docs**:

1. `medquad_ans_0002833-4` (score=0.0328) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is...
2. `medquad_ans_0000465-5` (score=0.0161) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - MedlinePlus Encyclope...
3. `medquad_ans_0000785-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0002833-3` (score=0.0159) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an enzyme called fumar...
5. `medquad_ans_0000528-4` (score=0.0159) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4].

**Citations detected**: `medquad_ans_0002833-4`
**Supported citations**: `medquad_ans_0002833-4`
**Unsupported citations**: _none_

## Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.0315) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the table below includes how...
2. `medquad_ans_0004193-1` (score=0.0276) - What are the signs and symptoms of Multiple epiphyseal dysplasia 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple epiphyseal dysplasia 4. If the information is available, the table below incl...
3. `medquad_ans_0005857-1` (score=0.0245) - What are the signs and symptoms of Spondylometaphyseal dysplasia X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia X-linked. If the information is available, the ta...
4. `medquad_ans_0005848-1` (score=0.0226) - What are the signs and symptoms of Spondylometaphyseal dysplasia Algerian type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia Algerian type. If the information is availab...
5. `medquad_ans_0000389-1` (score=0.0164) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also includes otopalat...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Top docs**:

1. `medquad_ans_0000097-11` (score=0.0290) - - Diverticular disease is a condition that occurs when a person has problems from small pouches, or sacs, that have formed and pushed outward through weak spots in the colon wall. The problems that occur with diverticular disease include...
2. `medquad_ans_0000128-7` (score=0.0287) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding, you need to go to t...
3. `medquad_ans_0000097-7` (score=0.0243) - Diverticulosis Health care providers often find diverticulosis during a routine x ray or a colonoscopy, a test used to look inside the rectum and entire colon to screen for colon cancer or polyps or to evaluate the source of rectal bleed...
4. `medquad_ans_0000126-7` (score=0.0235) - Treatment for Crohns disease depends on - where the disease is located in the GI tract - what problems you already have from the disease - what past treatments you have had for the disease The goals of treatment are to - decrease the inf...
5. `medquad_ans_0000038-22` (score=0.0164) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and mark those that are m...

**Answer**:

The treatments for diverticular disease depend on the severity of the condition and whether the person has diverticulosis or diverticulitis. For diverticulosis with mild symptoms, health care providers may recommend a high-fiber diet, fiber supplements, medications, and possibly probiotics [medquad_ans_0000097-11].

For diverticulitis with mild symptoms and no complications, treatment typically includes rest, oral antibiotics, and a liquid diet for a period of time [medquad_ans_0000097-11]. If diverticulitis is severe or leads to complications such as an abscess, perforation, peritonitis, fistula, or intestinal obstruction, these require medical intervention to prevent worsening and serious illness [medquad_ans_0000097-11].

In cases of diverticular bleeding, which is rare, bleeding may stop on

**Citations detected**: `medquad_ans_0000097-11`
**Supported citations**: `medquad_ans_0000097-11`
**Unsupported citations**: _none_

## Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Top docs**:

1. `medquad_ans_0006066-1` (score=0.0325) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0000619-1__dup1` (score=0.0323) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is available, the table belo...
3. `medquad_ans_0000621-1__dup1` (score=0.0161) - What are the signs and symptoms of Baraitser-Winter syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser-Winter syndrome. If the information is available, the table below includes how oft...
4. `medquad_ans_0000108-2__dup1` (score=0.0159) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
5. `medquad_ans_0005488-1` (score=0.0156) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how often the symptom is ...

**Answer**:

The provided context does not mention "Baraitser Brett Piesowicz syndrome" or any condition with a similar name. Therefore, based on the given information, the symptoms of Baraitser Brett Piesowicz syndrome cannot be determined.

Note: The context discusses Thompson Baraitser syndrome, not Baraitser Brett Piesowicz syndrome. There is no evidence in the provided text to support the existence or symptoms of the latter condition. [medquad_ans_0006066-1]

**Citations detected**: `medquad_ans_0006066-1`
**Supported citations**: `medquad_ans_0006066-1`
**Unsupported citations**: _none_

## Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Top docs**:

1. `medquad_ans_0003911-2` (score=0.0328) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a meningioma. When treatm...
2. `medquad_ans_0000006_3-8` (score=0.0190) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
3. `medquad_ans_0003911-1` (score=0.0161) - Meningiomas originate in the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign, though a minority of meningiomas can be classified as atypical or malignant. Though rare, malignant meningiomas ca...
4. `medquad_ans_0000052-2__dup1` (score=0.0161) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
5. `medquad_ans_0003103-4` (score=0.0159) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moisturize) the skin, h...

**Answer**:

The treatments for meningioma vary depending on the location of the tumor and the symptoms it causes. Careful observation may be the best course of action for some individuals. When treatment is necessary, surgery and radiation are the most common options. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. It is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs, which prevent the release of growth hormones. Additionally, new treatments such as lenalidomide, an angiogenesis inhibitor that prevents the growth of new blood vessels needed by tumors, are being tested in clinical trials. Clinical trials are an important part of

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.0320) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a team of health care pr...
2. `medquad_ans_0000006_6-8` (score=0.0296) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health care providers who ...
3. `medquad_ans_0000006_7-2` (score=0.0295) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Staging ...
4. `medquad_ans_0000006_7-7` (score=0.0294) - Check the list of NCI-supported cancer clinical trials that are now accepting patients with childhood central nervous system germ cell tumor. For more specific results, refine the search by using other search features, such as the locati...
5. `medquad_ans_0000015_1-8` (score=0.0283) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers who are experts in ...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials for new treatments. Standard treatments are currently used, and some treatments are being tested in clinical trials. Treatment planning should be done by a team of health care providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8].

**Citations detected**: `medquad_ans_0000006_7-8`
**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

## Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Top docs**:

1. `medquad_ans_0004401-3` (score=0.0320) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose triglyceride lipase (ATGL...
2. `medquad_ans_0000729-2` (score=0.0320) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
3. `medquad_ans_0004401-4` (score=0.0318) - How is neutral lipid storage disease with myopathy inherited? This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal ...
4. `medquad_ans_0004401-1` (score=0.0311) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condit...
5. `medquad_ans_0000729-1` (score=0.0308) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the accumulation of fats...

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene, which impairs the function of the adipose triglyceride lipase (ATGL) enzyme, resulting in the inability to break down triglycerides into fatty acids for energy use. As a result, triglycerides accumulate in muscle and other tissues. The condition is inherited in an autosomal recessive pattern, meaning both copies of the PNPLA2 gene in each cell are mutated, and affected individuals typically have parents who carry one mutated copy but do not show symptoms. Other features may include a fatty liver, cardiomyopathy, pancreatitis, hypothyroidism, and type 2 diabetes mellitus. The signs and symptoms vary greatly among affected individuals,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.0328) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. Cases of Zika virus ...
2. `medquad_ans_0006546-2` (score=0.0320) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, drink fluids, and take...
3. `medquad_ans_0000981-1__dup1` (score=0.0320) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus h...
4. `medquad_ans_0000174-1__dup3` (score=0.0310) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected blood. There have b...
5. `medquad_ans_0000193-1__dup2` (score=0.0308) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may develop in the first fe...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and via blood transfusions. The illness associated with Zika virus infection is usually mild, with common symptoms including fever, rash, joint pain, and conjunctivitis (red eyes). Recent research has suggested a link between Zika virus infection and Guillain-Barre syndrome (GBS), and prenatal infection has been shown to cause adverse pregnancy outcomes such as microcephaly and other serious brain anomalies [medquad_ans_0006546-1]. There is no vaccine or specific medicine to treat Zika virus; management includes rest, hydration, and use of acetaminophen for pain, with avoidance of aspirin and NSAIDs until dengue is ruled out [medquad_ans_00065

**Citations detected**: `medquad_ans_0006546-1`
**Supported citations**: `medquad_ans_0006546-1`
**Unsupported citations**: _none_

## Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=0.0323) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a heart attack, a more ...
2. `medquad_ans_0000019-3__dup2` (score=0.0323) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack, what is happening ...
3. `medquad_ans_0000019-4__dup2` (score=0.0323) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not apply to broken heart sy...
4. `medquad_ans_0000019-7` (score=0.0310) - Even though broken heart syndrome may feel like a heart attack, its a very different problem that needs a different type of treatment. The good news is that broken heart syndrome is usually treatable, and most people make a full recovery...
5. `medquad_ans_0000019-5__dup1` (score=0.0303) - All chest pain should be checked by a doctor. Because symptoms of broken heart syndrome are similar to those of a heart attack, it is important to seek help right away. Your doctor may not be able to diagnose broken heart syndrome until ...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Citations detected**: `medquad_ans_0000019-1__dup2`
**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

## Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Top docs**:

1. `medquad_ans_0000448-3` (score=0.0318) - Hashimoto thyroiditis is thought to result from a combination of genetic and environmental factors. Some of these factors have been identified, but many remain unknown. Hashimoto thyroiditis is classified as an autoimmune disorder, one o...
2. `medquad_ans_0000448-2` (score=0.0318) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid underactivity (hypothyroi...
3. `medquad_ans_0000448-1` (score=0.0304) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functions. For example, thy...
4. `medquad_ans_0000448-4` (score=0.0301) - The inheritance pattern of Hashimoto thyroiditis is unclear because many genetic and environmental factors appear to be involved. However, the condition can cluster in families, and having a close relative with Hashimoto thyroiditis or a...
5. `medquad_ans_0000009-3__dup4` (score=0.0290) - Hypothyroidism has several causes, including - Hashimotos disease - thyroiditis, or inflammation of the thyroid - congenital hypothyroidism, or hypothyroidism that is present at birth - surgical removal of part or all of the thyroid - ra...

**Answer**:

Hashimoto thyroiditis is an autoimmune disorder in which the immune system attacks the thyroid gland, leading to damage of thyroid cells and a subsequent shortage of thyroid hormones. This occurs when white blood cells called lymphocytes accumulate abnormally in the thyroid and produce antibodies that destroy thyroid cells. The resulting hormonal deficiency underlies the signs and symptoms of the condition. It is the most common cause of hypothyroidism in the United States and affects 1 to 2 percent of people, occurring more frequently in women than in men, possibly due to hormonal factors. Genetic and environmental factors contribute to its development, including variations in genes within the human leukocyte antigen (HLA) complex and immune-regulating genes, as well as potential triggers such as viral infections, changes in sex hormones, medications, ionizing radiation, and excess iodine intake. People with Hashimoto thyroiditis also have an increased risk of developing other autoimmune disorders, such as vitiligo, rheumat

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Top docs**:

1. `medquad_ans_0001043-2` (score=0.0325) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
2. `medquad_ans_0006463-1` (score=0.0320) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include enlarged liver and ...
3. `medquad_ans_0006463-2` (score=0.0313) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often the symptom is seen...
4. `medquad_ans_0001043-1` (score=0.0308) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, bone marrow, small i...
5. `medquad_ans_0001043-3` (score=0.0303) - Mutations in the LIPA gene cause Wolman disease. The LIPA gene provides instructions for producing an enzyme called lysosomal acid lipase. This enzyme is found in the lysosomes (compartments that digest and recycle materials in the cell)...

**Answer**:

Wolman disease is a type of lysosomal storage disorder that is an inherited condition causing a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. It is caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. Common symptoms in infants include enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients from food. The condition is severe and life-threatening, though new therapies such as bone marrow transplantation and enzyme replacement therapy are showing promise in improving outcomes [medquad_ans_0006463-1]. It is estimated to occur in 1 in 350,000 newborns [medquad_ans_0001043-2].

**Citations detected**: `medquad_ans_0006463-1`, `medquad_ans_0001043-2`
**Supported citations**: `medquad_ans_0006463-1`, `medquad_ans_0001043-2`
**Unsupported citations**: _none_

## Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.0325) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational viny...
2. `medquad_ans_0002753-1` (score=0.0323) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors are typically slow-...
3. `medquad_ans_0000765-3__dup1` (score=0.0214) - What causes blue rubber bleb nevus syndrome? Currently the cause of blue rubber bleb syndrome is not known.
4. `medquad_ans_0001853-3` (score=0.0161) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...
5. `medquad_ans_0002753-2` (score=0.0159) - What are the signs and symptoms of Hemangiopericytoma? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemangiopericytoma. If the information is available, the table below includes how often the symptom...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Citations detected**: `medquad_ans_0002753-3`
**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

## Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Top docs**:

1. `medquad_ans_0000593-3` (score=0.0323) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a protein called the B...
2. `medquad_ans_0000655-3` (score=0.0320) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or more genes. Most ge...
3. `medquad_ans_0000655-1` (score=0.0276) - Microphthalmia is an eye abnormality that arises before birth. In this condition, one or both eyeballs are abnormally small. In some affected individuals, the eyeball may appear to be completely missing; however, even in these cases some...
4. `medquad_ans_0000743-3` (score=0.0260) - Ocular albinism type 1 results from mutations in the GPR143 gene. This gene provides instructions for making a protein that plays a role in pigmentation of the eyes and skin. It helps control the growth of melanosomes, which are cellular...
5. `medquad_ans_0003520-3` (score=0.0161) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. These genes are contai...

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. In specific cases, such as Lenz microphthalmia syndrome, mutations in at least two genes on the X chromosome are thought to be responsible, with only the BCOR gene currently identified. The BCOR gene provides instructions for making a protein called the BCL6 corepressor, which helps regulate the activity of other genes and plays an important role in early embryonic development. A mutation in the BCOR gene disrupts normal development of the eyes and other organs before birth [medquad_ans_0000593-3]. Additionally, microphthalmia may result from chromosomal abnormalities affecting one or more genes [medquad_ans_0000655-3]. Most genetic changes associated with isolated microphthalmia have been identified in only a small number of affected individuals [med

**Citations detected**: `medquad_ans_0000593-3`, `medquad_ans_0000655-3`
**Supported citations**: `medquad_ans_0000593-3`, `medquad_ans_0000655-3`
**Unsupported citations**: _none_

## Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Top docs**:

1. `medquad_ans_0002277-2` (score=0.0289) - What are the signs and symptoms of Familial juvenile hyperuricaemic nephropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial juvenile hyperuricaemic nephropathy. If the information is avail...
2. `medquad_ans_0003550-2` (score=0.0278) - What are the signs and symptoms of Lesch Nyhan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lesch Nyhan syndrome. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0003870-2` (score=0.0258) - What are the signs and symptoms of Medullary cystic kidney disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Medullary cystic kidney disease. If the information is available, the table below incl...
4. `medquad_ans_0003871-2` (score=0.0257) - What are the signs and symptoms of Medullary cystic kidney disease 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Medullary cystic kidney disease 1. If the information is available, the table below ...
5. `medquad_ans_0003833-1` (score=0.0254) - What are the signs and symptoms of Maturity-onset diabetes of the young, type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Maturity-onset diabetes of the young, type 5. If the information is avail...

**Answer**:

Gout is listed as a symptom in Familial juvenile hyperuricaemic nephropathy and Lesch Nyhan syndrome. In Familial juvenile hyperuricaemic nephropathy, gout is mentioned as one of the signs and symptoms [medquad_ans_0002277-2]. In Lesch Nyhan syndrome, gout (specifically in the feet) is noted as a symptom occurring in 90% of patients [medquad_ans_0003550-2].

Thus, the symptoms of gout in these conditions include gout, particularly in the feet in Lesch Nyhan syndrome. [medquad_ans_0002277-2][medquad_ans_0003550-2]

**Citations detected**: `medquad_ans_0002277-2`, `medquad_ans_0003550-2`
**Supported citations**: `medquad_ans_0002277-2`, `medquad_ans_0003550-2`
**Unsupported citations**: _none_

## Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Top docs**:

1. `medquad_ans_0000307-1__dup1` (score=0.0304) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a group of muscles or ...
2. `medquad_ans_0000859-1` (score=0.0301) - Rapid-onset dystonia parkinsonism is a rare movement disorder. "Rapid-onset" refers to the abrupt appearance of signs and symptoms over a period of hours to days. Dystonia is a condition characterized by involuntary, sustained muscle con...
3. `medquad_ans_0000309-1` (score=0.0271) - Dystonia 6 is one of many forms of dystonia, which is a group of conditions characterized by involuntary movements, twisting (torsion) and tensing of various muscles, and unusual positioning of affected body parts. Dystonia 6 can appear ...
4. `medquad_ans_0001999-1` (score=0.0270) - What are the signs and symptoms of Dystonia 6, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 6, torsion. If the information is available, the table below includes how often the sympt...
5. `medquad_ans_0002000-1` (score=0.0264) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes how often the sympt...

**Answer**:

The provided context does not mention Dystonia 8 or its causes. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.0318) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unknown.
2. `medquad_ans_0000573-2` (score=0.0249) - Lactose intolerance in infancy resulting from congenital lactase deficiency is a rare disorder. Its incidence is unknown. This condition is most common in Finland, where it affects an estimated 1 in 60,000 newborns. Approximately 65 perc...
3. `medquad_ans_0000612-1` (score=0.0164) - Lysinuric protein intolerance is a disorder caused by the body's inability to digest and use certain protein building blocks (amino acids), namely lysine, arginine, and ornithine. Because the body cannot effectively break down these amin...
4. `medquad_ans_0003719-1` (score=0.0161) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids, which are found in ...
5. `medquad_ans_0000650-2` (score=0.0161) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Citations detected**: `medquad_ans_0000612-2`
**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

## Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Top docs**:

1. `medquad_ans_0005876-1` (score=0.0290) - What are the signs and symptoms of STAR syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for STAR syndrome. If the information is available, the table below includes how often the symptom is seen i...
2. `medquad_ans_0004850-1` (score=0.0273) - What are the signs and symptoms of Pheochromocytoma, childhood? The Human Phenotype Ontology provides the following list of signs and symptoms for Pheochromocytoma, childhood. If the information is available, the table below includes how...
3. `medquad_ans_0002728-2` (score=0.0254) - What are the signs and symptoms of Hardikar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hardikar syndrome. If the information is available, the table below includes how often the symptom i...
4. `medquad_ans_0005294-1` (score=0.0228) - What are the signs and symptoms of Renal dysplasia-limb defects syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Renal dysplasia-limb defects syndrome. If the information is available, the tabl...
5. `medquad_ans_0004936-2` (score=0.0227) - What are the signs and symptoms of Polyarteritis nodosa? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyarteritis nodosa. If the information is available, the table below includes how often the sym...

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. [medquad_ans_0005876-1] and [medquad_ans_0004850-1] discuss symptoms of STAR syndrome and Pheochromocytoma, childhood, respectively, but neither addresses Renal Artery Stenosis. Therefore, the requested information is not available in the context.

**Citations detected**: `medquad_ans_0005876-1`, `medquad_ans_0004850-1`
**Supported citations**: `medquad_ans_0005876-1`, `medquad_ans_0004850-1`
**Unsupported citations**: _none_

## Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.0325) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis - Genetic Testing Reg...
2. `medquad_ans_0000059-16` (score=0.0236) - There are many types of treatments. Many are medicines and other treatments your doctor will have to prescribe. But there are other types of treatments you can buy without a prescription or try on your own. Some treatments for psoriasis ...
3. `medquad_ans_0003103-4` (score=0.0235) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moisturize) the skin, h...
4. `medquad_ans_0004909-4` (score=0.0205) - How might pityriasis rubra pilaris be treated? Treatment of pityriasis rubra pilaris (PRP) is mainly based on reports of patients' experiences. No controlled trials have been done, so the effectiveness and safety of treatments is unclear...
5. `medquad_ans_0000021-1` (score=0.0164) - Adermatoglyphia is the absence of ridges on the skin on the pads of the fingers and toes, as well as on the palms of the hands and soles of the feet. The patterns of these ridges (called dermatoglyphs) form whorls, arches, and loops that...

**Answer**:

The provided context does not contain specific information about the treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While the context lists resources that address the diagnosis or management of these conditions, such as the Genetic Testing Registry and MedlinePlus, it does not detail any available treatments [medquad_ans_0000712-5]. Therefore, based on the given information, the treatments for these syndromes cannot be determined.

**Citations detected**: `medquad_ans_0000712-5`
**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

## Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.0328) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is available, the tabl...
2. `medquad_ans_0000648-1__dup1` (score=0.0283) - What are the signs and symptoms of Bartter syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Bartter syndrome type 3. If the information is available, the table below includes how often t...
3. `medquad_ans_0000589-2__dup1` (score=0.0279) - What are the signs and symptoms of Autosomal recessive pseudohypoaldosteronism type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive pseudohypoaldosteronism type 1. If the informat...
4. `medquad_ans_0002593-2` (score=0.0255) - What are the signs and symptoms of Glucocorticoid-remediable aldosteronism? The Human Phenotype Ontology provides the following list of signs and symptoms for Glucocorticoid-remediable aldosteronism. If the information is available, the ...
5. `medquad_ans_0002271-1` (score=0.0231) - What are the signs and symptoms of Familial hypocalciuric hypercalcemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hypocalciuric hypercalcemia type 3. If the information is availab...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Citations detected**: `medquad_ans_0002259-1`
**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

## Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Top docs**:

1. `medquad_ans_0005761-3` (score=0.0164) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation. Prenatal...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0005761-1` (score=0.0161) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal features (increased...
4. `medquad_ans_0000642-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0005777-1` (score=0.0159) - Spinocerebellar ataxia 3 is a rare movement disorder that is characterized by ataxia, clumsiness and weakness in the arms and legs, spasticity, a staggering lurching walk easily mistaken for drunkenness, difficulty with speech and swallo...

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Citations detected**: `medquad_ans_0005761-3`
**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

## Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.0318) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload. If the information ...
2. `medquad_ans_0000340-1__dup1` (score=0.0290) - What are the signs and symptoms of Anemia sideroblastic and spinocerebellar ataxia? The Human Phenotype Ontology provides the following list of signs and symptoms for Anemia sideroblastic and spinocerebellar ataxia. If the information is...
3. `medquad_ans_0006525-2` (score=0.0287) - What are the signs and symptoms of X-linked sideroblastic anemia? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked sideroblastic anemia. If the information is available, the table below includes...
4. `medquad_ans_0000258-2__dup1` (score=0.0286) - What are the signs and symptoms of Alpha-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0000706-2__dup1` (score=0.0265) - What are the signs and symptoms of Beta-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta-thalassemia. If the information is available, the table below includes how often the symptom is ...

**Answer**:

The signs and symptoms of Hypochromic microcytic anemia with iron overload include abnormality of metabolism/homeostasis, abnormality of the liver, anemia, and autosomal recessive inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. Specific frequencies for individual symptoms are not provided in the context. [medquad_ans_0003040-1]

**Citations detected**: `medquad_ans_0003040-1`
**Supported citations**: `medquad_ans_0003040-1`
**Unsupported citations**: _none_

## Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Top docs**:

1. `medquad_ans_0000115-2` (score=0.0164) - Benign chronic pemphigus is a rare condition; its prevalence is unknown.
2. `medquad_ans_0000441-4` (score=0.0164) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family members with the con...
3. `medquad_ans_0000115-1` (score=0.0161) - Benign chronic pemphigus, often called Hailey-Hailey disease, is a rare skin condition that usually appears in early adulthood. The disorder is characterized by red, raw, and blistered areas of skin that occur most often in skin folds, s...
4. `medquad_ans_0000785-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000115-5` (score=0.0159) - These resources address the diagnosis or management of benign chronic pemphigus: - American Osteopathic College of Dermatology - Genetic Testing Registry: Familial benign pemphigus These resources from MedlinePlus offer information about...

**Answer**:

Benign chronic pemphigus is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family members with the condition [medquad_ans_0000441-4].

Note: While some sources mention autosomal recessive inheritance, the context specifically identifies benign chronic pemphigus as being inherited in an autosomal dominant pattern [medquad_ans_0000441-4]. The autosomal recessive pattern is described in a different condition, not benign chronic pemphigus. Therefore, the correct inheritance pattern for benign chronic pemphigus is autosomal dominant.

**Citations detected**: `medquad_ans_0000441-4`
**Supported citations**: `medquad_ans_0000441-4`
**Unsupported citations**: _none_

## Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Top docs**:

1. `medquad_ans_0000916-1__dup1` (score=0.0164) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
2. `medquad_ans_0000926-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000915-1__dup1` (score=0.0161) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
4. `medquad_ans_0000424-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000916-5__dup1` (score=0.0159) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene known to be associa...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant fashion. [medquad_ans_0000916-1__dup1]

**Citations detected**: `medquad_ans_0000916-1__dup1`
**Supported citations**: `medquad_ans_0000916-1__dup1`
**Unsupported citations**: _none_

## Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Top docs**:

1. `medquad_ans_0000483-3` (score=0.0320) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family of proteins that tran...
2. `medquad_ans_0005069-3` (score=0.0219) - What genes are related to Hutchinson-Gilford progeria syndrome? Mutations in the LMNA gene cause Hutchinson-Gilford progeria syndrome. The LMNA gene provides instructions for making a protein called lamin A. This protein plays an importa...
3. `medquad_ans_0000483-5` (score=0.0164) - These resources address the diagnosis or management of histiocytosis-lymphadenopathy plus syndrome: - Genetic Testing Registry: Histiocytosis-lymphadenopathy plus syndrome These resources from MedlinePlus offer information about the diag...
4. `medquad_ans_0000580-3` (score=0.0164) - Somatic mutations in the BRAF gene have been identified in the Langerhans cells of about half of individuals with Langerhans cell histiocytosis. Somatic gene mutations are acquired during a person's lifetime and are present only in certa...
5. `medquad_ans_0002895-1` (score=0.0161) - Histiocytosis-lymphadenopathy plus syndrome is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H syndrome, pigmented hypertrichosis with insulin-dependent dia...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0000483-3]. These mutations reduce or eliminate the activity of the equilibrative nucleoside transporter 3 (ENT3) protein, which is responsible for transporting nucleosides out of lysosomes and into mitochondria [medquad_ans_0000483-3]. This impairment in nucleoside transport is believed to lead to a buildup of nucleosides in lysosomes and a shortage in mitochondria, potentially affecting cellular energy production and contributing to the condition's symptoms [medquad_ans_0000483-3].

**Citations detected**: `medquad_ans_0000483-3`
**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

## Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Top docs**:

1. `medquad_ans_0004453-1` (score=0.0164) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0004454-1` (score=0.0161) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
4. `medquad_ans_0000647-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0004456-1` (score=0.0159) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, but many cases are due to a new mutation and are not inherited from an affected parent [medquad_ans_0004453-1].

**Citations detected**: `medquad_ans_0004453-1`
**Supported citations**: `medquad_ans_0004453-1`
**Unsupported citations**: _none_

## Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.0318) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Acidosis These resource...
2. `medquad_ans_0000396-5` (score=0.0271) - These resources address the diagnosis or management of fumarase deficiency: - Gene Review: Gene Review: Fumarate Hydratase Deficiency - Genetic Testing Registry: Fumarase deficiency These resources from MedlinePlus offer information abou...
3. `medquad_ans_0001093-1` (score=0.0164) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental and motor skills (psy...
4. `medquad_ans_0001089-5` (score=0.0164) - These resources address the diagnosis or management of HMG-CoA lyase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of hydroxymethylglutaryl-CoA lyase These resources from MedlinePlus offer information about the d...
5. `medquad_ans_0001093-3` (score=0.0161) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein building block (amino acid...

**Answer**:

The provided context does not mention any specific treatments for 3-methylglutaconyl-CoA hydratase deficiency. It describes diagnostic features, such as elevated levels of 3-methylglutaconic acid in urine and metabolic acidosis, and outlines neurological symptoms, but does not address therapeutic interventions [medquad_ans_0001093-1]. Therefore, based on the given information, the treatments for this condition cannot be determined.

**Citations detected**: `medquad_ans_0001093-1`
**Supported citations**: `medquad_ans_0001093-1`
**Unsupported citations**: _none_

## Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.0325) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0003000-2` (score=0.0318) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, the table below incl...
3. `medquad_ans_0000494-1` (score=0.0310) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the...
4. `medquad_ans_0000126-1__dup1` (score=0.0310) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affected muscles become sl...
5. `medquad_ans_0000494-5` (score=0.0301) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis - Genetic Testing Re...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although muscle atrophy or hypertrophy may occur [medquad_ans_0000494-1]. The condition affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Citations detected**: `medquad_ans_0000494-1`, `medquad_ans_0000494-2`
**Supported citations**: `medquad_ans_0000494-1`, `medquad_ans_0000494-2`
**Unsupported citations**: _none_

## Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.0325) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of extraocular muscles...
2. `medquad_ans_0000084-2__dup1` (score=0.0272) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ephedrine, and 3,4-d...
3. `medquad_ans_0000225-1` (score=0.0257) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Congenital fibrosis of t...
4. `medquad_ans_0000228-5` (score=0.0161) - These resources address the diagnosis or management of congenital hepatic fibrosis: - Gene Review: Gene Review: Congenital Hepatic Fibrosis Overview - Genetic Testing Registry: Congenital hepatic fibrosis These resources from MedlinePlus...
5. `medquad_ans_0000225-4` (score=0.0159) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suff...

**Answer**:

The provided context does not contain information about treatments for congenital fibrosis of the extraocular muscles. Therefore, based on the available resources, no treatments can be identified. [medquad_ans_0000225-5] and [medquad_ans_0000225-1] describe the diagnosis, symptoms, and genetic subtypes of congenital fibrosis of the extraocular muscles, but do not mention specific treatments. [medquad_ans_0000084-2__dup1] discusses treatments for congenital myasthenia, which is a different condition.

Thus, the context is insufficient to answer the question about treatments for congenital fibrosis of the extraocular muscles.

**Citations detected**: `medquad_ans_0000225-5`, `medquad_ans_0000225-1`, `medquad_ans_0000084-2__dup1`
**Supported citations**: `medquad_ans_0000225-5`, `medquad_ans_0000225-1`, `medquad_ans_0000084-2__dup1`
**Unsupported citations**: _none_

## Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.0325) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms of HD may include u...
2. `medquad_ans_0000075-3__dup1` (score=0.0299) - The prognosis for individuals with chorea varies depending on the type of chorea and the associated disease. Huntington's disease is a progressive, and ultimately, fatal disease. Syndenham's chorea is treatable and curable.
3. `medquad_ans_0000257-1__dup3` (score=0.0280) - Degenerative nerve diseases affect many of your body's activities, such as balance, movement, talking, breathing, and heart function. Many of these diseases are genetic. Sometimes the cause is a medical condition such as alcoholism, a tu...
4. `medquad_ans_0000152-1__dup1` (score=0.0278) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressively worse, include u...
5. `medquad_ans_0000013-9__dup1` (score=0.0277) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples of heritable diseases...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. It is a progressive and ultimately fatal disease, with symptoms that typically begin between ages 30 and 50, although an earlier onset form called juvenile HD can occur under age 20 [medquad_ans_0000152-1__dup1]. Early symptoms may include uncontrolled movements (called chorea), abnormal body postures, changes in behavior, emotion, judgment, and cognition, impaired coordination, slurred speech, and difficulty feeding and swallowing [medquad_ans_0000491-1__dup2]. People are born with the defective gene, and if one parent has HD, there is a 50 percent chance of inheriting it. There is no cure; medicines can help manage symptoms but cannot

**Citations detected**: `medquad_ans_0000152-1__dup1`, `medquad_ans_0000491-1__dup2`
**Supported citations**: `medquad_ans_0000152-1__dup1`, `medquad_ans_0000491-1__dup2`
**Unsupported citations**: _none_

## Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.0323) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
2. `medquad_ans_0000416-3` (score=0.0164) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
3. `medquad_ans_0000416-1` (score=0.0161) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with the mild form of glu...
4. `medquad_ans_0000359-2` (score=0.0161) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000402-2` (score=0.0159) - The prevalence of GACI has been estimated to be about 1 in 391,000. At least 200 affected individuals have been described in the medical literature.

**Answer**:

The number of people affected by glutamate formiminotransferase deficiency is not precisely defined in the provided context. However, it is noted that approximately 20 affected individuals have been identified, and the condition affects about 1 per million people worldwide [medquad_ans_0000359-2]. Additionally, the prevalence of GACI (which may be related to this condition) is estimated at about 1 in 391,000 [medquad_ans_0000402-2]. Since the context does not provide a single, definitive global number, the most accurate answer based on available information is that the condition affects about 1 per million people worldwide.

[medquad_ans_0000359-2] [medquad_ans_0000402-2]

**Citations detected**: `medquad_ans_0000359-2`, `medquad_ans_0000402-2`
**Supported citations**: `medquad_ans_0000359-2`, `medquad_ans_0000402-2`
**Unsupported citations**: _none_

## Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.0279) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0004450-1` (score=0.0164) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
3. `medquad_ans_0004452-1` (score=0.0161) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
4. `medquad_ans_0000359-2` (score=0.0161) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0004456-1` (score=0.0159) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2]. Additionally, it affects about 1 per million people worldwide [medquad_ans_000359-2]. These estimates provide a range of prevalence for the condition.

**Citations detected**: `medquad_ans_0000738-2`, `medquad_ans_000359-2`
**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: `medquad_ans_000359-2`

## Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.0328) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, e...
2. `medquad_ans_0000339-1__dup3` (score=0.0315) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels of society.
3. `medquad_ans_0000339-2__dup1` (score=0.0312) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasionally pubic lice may be...
4. `medquad_ans_0000557-1__dup2` (score=0.0310) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice that live on humans a...
5. `medquad_ans_0000339-7` (score=0.0308) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to help prevent and con...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Citations detected**: `medquad_ans_0000339-5__dup1`
**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

## Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.0325) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of the disease, and other...
2. `medquad_ans_0005571-2` (score=0.0164) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has been a documented c...
3. `medquad_ans_0000803-5` (score=0.0161) - These resources address the diagnosis or management of Poland syndrome: - Children's Medical Center of Dallas - Great Ormond Street Hospital (UK): Treatment Options for Symbrachydactyly - St. Louis Children's Hospital: Chest Wall Deformi...
4. `medquad_ans_0005571-1` (score=0.0159) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration and frequency of the...
5. `medquad_ans_0000901-5` (score=0.0159) - These resources address the diagnosis or management of Sheldon-Hall syndrome: - Gillette Children's Hospital - NYU Langone Medical Center These resources from MedlinePlus offer information about the diagnosis and management of various he...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. The effectiveness of these treatments varies among individuals. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Citations detected**: `medquad_ans_0005571-3`
**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

## Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.0320) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People with FG syndrome frequ...
2. `medquad_ans_0000373-1` (score=0.0318) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects intelligence and beh...
3. `medquad_ans_0002362-1` (score=0.0310) - What are the signs and symptoms of FG syndrome 4? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome 4. If the information is available, the table below includes how often the symptom is seen i...
4. `medquad_ans_0000373-2` (score=0.0308) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen with other disorders.
5. `medquad_ans_0000373-5` (score=0.0308) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testing Registry: FG synd...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Citations detected**: `medquad_ans_0002359-1`
**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

## Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.0325) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often the symptom is seen...
2. `medquad_ans_0000225-1` (score=0.0164) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Congenital fibrosis of t...
3. `medquad_ans_0006572-2` (score=0.0161) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table below includes how oft...
4. `medquad_ans_0000225-4` (score=0.0159) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suff...
5. `medquad_ans_0003554-1` (score=0.0159) - What are the signs and symptoms of Lethal congenital contracture syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Lethal congenital contracture syndrome 1. If the information is available, th...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, and syndactyly [medquad_ans_0006242-1]. Congenital fibrosis of the extraocular muscles leads to impaired eye movement, difficulty looking upward, limited side-to-side eye movement, eye misalignment (strabismus), and droopy eyelids (ptosis), which can further limit vision [medquad_ans_0000225-1]. Tukel syndrome is also characterized by missing fingers (oligodactyly) [medquad_ans_0000225-1].

**Citations detected**: `medquad_ans_0006242-1`, `medquad_ans_0000225-1`
**Supported citations**: `medquad_ans_0006242-1`, `medquad_ans_0000225-1`
**Unsupported citations**: _none_

## Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Top docs**:

1. `medquad_ans_0004155-5` (score=0.0216) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missing IDUA enzyme. A dru...
2. `medquad_ans_0000199-1__dup1` (score=0.0164) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty materials (lipids) ac...
3. `medquad_ans_0000676-5` (score=0.0164) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloudy Cornea - MedlinePl...
4. `medquad_ans_0003103-4` (score=0.0161) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moisturize) the skin, h...
5. `medquad_ans_0000677-5` (score=0.0161) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastrin These resources f...

**Answer**:

The provided context does not contain information about the treatments for mucolipidoses. While it describes the genetic basis and pathophysiology of mucolipidoses, including the accumulation of carbohydrates and lipids due to enzyme deficiencies, no specific treatments are mentioned [medquad_ans_0000199-1__dup1]. Therefore, based on the available context, the treatments for mucolipidoses cannot be determined.

**Citations detected**: `medquad_ans_0000199-1__dup1`
**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

## Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.0325) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis type 2. If the info...
2. `medquad_ans_0001157-1` (score=0.0310) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4. If the information ...
3. `medquad_ans_0003240-2` (score=0.0302) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is available, the tabl...
4. `medquad_ans_0004334-2` (score=0.0277) - What are the signs and symptoms of Neonatal intrahepatic cholestasis caused by citrin deficiency? Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is characterized by transient intrahepatic cholestasis, diffuse fatty...
5. `medquad_ans_0000689-1__dup1` (score=0.0252) - What are the signs and symptoms of Benign recurrent intrahepatic cholestasis? The Human Phenotype Ontology provides the following list of signs and symptoms for Benign recurrent intrahepatic cholestasis. If the information is available, ...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Citations detected**: `medquad_ans_0005084-2`
**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

## Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.0318) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he is more likely to dev...
2. `medquad_ans_0000070-7` (score=0.0286) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is help. Urinary incont...
3. `medquad_ans_0000070-8` (score=0.0281) - There are different types of urinary incontinence. Stress urinary incontinence happens when urine leaks as pressure is put on the bladder, for example, during exercise, coughing, sneezing, laughing, or lifting heavy objects. Its the most...
4. `medquad_ans_0000213-4__dup2` (score=0.0269) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 100 per 1,000 men.2 -...
5. `medquad_ans_0000931-1__dup2` (score=0.0246) - Urinary incontinence (UI) is loss of bladder control. Symptoms can range from mild leaking to uncontrollable wetting. It can happen to anyone, but it becomes more common with age. Women experience UI twice as often as men. Most bladder c...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6].

**Citations detected**: `medquad_ans_0000211-6`
**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

## Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0000882-1` (score=0.0317) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the severity of their sig...
2. `medquad_ans_0005415-1` (score=0.0315) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant Robinow syndrome. T...
3. `medquad_ans_0005414-1` (score=0.0306) - What are the signs and symptoms of Robinow Sorauf syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Robinow Sorauf syndrome. If the information is available, the table below includes how often t...
4. `medquad_ans_0000882-5` (score=0.0296) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Registry: Robinow syndro...
5. `medquad_ans_0005415-2` (score=0.0282) - What are the signs and symptoms of Robinow syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Robinow syndrome. If the information is available, the table below includes how often the symptom is ...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. It is characterized by two major types: autosomal recessive and autosomal dominant. Autosomal recessive Robinow syndrome is marked by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones leading to kyphoscoliosis, fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies." Other features may include underdeveloped genitalia, dental problems (such as crowded teeth and overgrown gums), kidney or heart defects, and delayed development in 10 to 15 percent of cases

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0000551-2` (score=0.0320) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
2. `medquad_ans_0003331-1` (score=0.0320) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
3. `medquad_ans_0003331-4` (score=0.0318) - How is juvenile primary lateral sclerosis inherited? Juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an...
4. `medquad_ans_0000551-1` (score=0.0313) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to motor neurons, which are...
5. `medquad_ans_0003331-3` (score=0.0311) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness (spasticity) of muscles in the arms, legs, and face. It is caused by mutations in the ALS2 gene, which leads to the production of an unstable or nonfunctional protein called alsin. This protein is abundant in motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. The damage to these motor neurons results in symptoms that begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness, and spasticity in the legs, along with difficulty with balance. As the condition progresses, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. The disorder is inherited in an autosomal recessive pattern, meaning both copies of the ALS2 gene in each cell have mutations, and each parent of an

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Top docs**:

1. `medquad_ans_0000040-1__dup3` (score=0.0301) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone Marrow Leukemia usual...
2. `medquad_ans_0000040-5__dup1` (score=0.0282) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment for leukemia is to de...
3. `medquad_ans_0000001_5-1` (score=0.0280) - Key Points - Hairy cell leukemia is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and platelets. - Gender and age may affect t...
4. `medquad_ans_0000001_3-1` (score=0.0272) - Key Points - Chronic lymphocytic leukemia is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and platelets. - Older age can affe...
5. `medquad_ans_0000554-1__dup2` (score=0.0269) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells crowd out the healthy...

**Answer**:

Leukemia is a cancer of the blood cells, specifically originating in the bone marrow, where it produces abnormal white blood cells called leukemia cells. These abnormal cells initially function almost normally but, over time, crowd out healthy blood cells, disrupting the normal functions of blood, such as fighting infection, carrying oxygen, and controlling bleeding. There are four common types of adult leukemia: chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic types progress slowly, while acute types worsen rapidly. Chronic lymphocytic leukemia is the most common type diagnosed in older adults, whereas acute lymphocytic leukemia is more common in children [medquad_ans_0000040-1__dup3].

**Citations detected**: `medquad_ans_0000040-1__dup3`
**Supported citations**: `medquad_ans_0000040-1__dup3`
**Unsupported citations**: _none_

## Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.0323) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common site is the temporal l...
2. `medquad_ans_0002485-2` (score=0.0315) - What are the signs and symptoms of gangliocytomas? Signs and symptoms caused by the presence of a gangliocytoma can vary depending on the tumor's location. Seizures are the most common symptom. Other symptoms may include increased brain ...
3. `medquad_ans_0000324-1__dup1` (score=0.0164) - Anaplastic astrocytoma is a rare, cancerous (malignant) type of brain tumor that arises from star-shaped brain cells called astrocytes. These cells surround and protect nerve cells in the brain and spinal cord. An anaplastic astrocytoma ...
4. `medquad_ans_0002574-1` (score=0.0161) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of glial cell is involved...
5. `medquad_ans_0001853-3` (score=0.0159) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most commonly affects individuals between the ages of 10 and 30. The most common site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy. Signs and symptoms vary depending on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. These tumors are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Citations detected**: `medquad_ans_0002485-1`
**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

## Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.0323) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of the tendon reflexes....
2. `medquad_ans_0000130-3__dup1` (score=0.0323) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are left with residual defi...
3. `medquad_ans_0000130-2__dup1` (score=0.0315) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportive care.
4. `medquad_ans_0000130-4__dup1` (score=0.0315) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
5. `medquad_ans_0000443-2` (score=0.0303) - The prevalence of Guillain-Barr syndrome is estimated to be 6 to 40 cases per 1 million people. The occurrence of the different types of Guillain-Barr syndrome varies across regions. AIDP is the most common type in North America and Euro...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barré syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barré syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1].

**Citations detected**: `medquad_ans_0000130-1__dup1`
**Supported citations**: `medquad_ans_0000130-1__dup1`
**Unsupported citations**: _none_

## Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.0323) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most pe...
2. `medquad_ans_0006570-4` (score=0.0315) - Is 22q11.2 deletion syndrome inherited? Most cases of 22q11.2 deletion syndrome are not inherited from a parent and are caused by a random error during the formation of egg or sperm cells, or during early fetal development. In about 10% ...
3. `medquad_ans_0001084-3` (score=0.0313) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have not been well charac...
4. `medquad_ans_0001084-1` (score=0.0306) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a location designated q...
5. `medquad_ans_0001084-4` (score=0.0291) - The inheritance of 22q11.2 deletion syndrome is considered autosomal dominant because a deletion in one copy of chromosome 22 in each cell is sufficient to cause the condition. Most cases of 22q11.2 deletion syndrome are not inherited, h...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell, specifically near the middle of the chromosome at a location designated q11.2. Most people with the syndrome are missing a piece that contains about 30 to 40 genes, many of which have not been well characterized. A small percentage of affected individuals have smaller deletions. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs, such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels. Loss of the TBX1 gene may also contribute to behavioral problems. The loss of another gene, COMT, may increase the risk of behavioral problems and mental illness. The other genes that are deleted likely contribute to the various features of the syndrome [medquad_ans_00065

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Top docs**:

1. `medquad_ans_0000812-1__dup2` (score=0.0323) - Shingles is a disease caused by the varicella-zoster virus - the same virus that causes chickenpox. After you have chickenpox, the virus stays in your body. It may not cause problems for many years. As you get older, the virus may reappe...
2. `medquad_ans_0000148-1__dup2` (score=0.0318) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain (which can be severe...
3. `medquad_ans_0000062-8` (score=0.0306) - Shingles is very common. Fifty percent of all Americans will have had shingles by the time they are 80. While shingles occurs in people of all ages, it is most common in 60-to 80-year-olds. In fact, one out of every three people 60 years...
4. `medquad_ans_0000062-1__dup5` (score=0.0299) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single stripe around either ...
5. `medquad_ans_0000062-2__dup3` (score=0.0295) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went away, but the virus r...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After a person has chickenpox, the virus remains in the body and can reactivate later in life, leading to shingles [medquad_ans_0000812-1__dup2].

**Citations detected**: `medquad_ans_0000812-1__dup2`
**Supported citations**: `medquad_ans_0000812-1__dup2`
**Unsupported citations**: _none_

## Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.0328) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the table below includes h...
2. `medquad_ans_0002200-2` (score=0.0289) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often the symptom is seen i...
3. `medquad_ans_0002458-2` (score=0.0264) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often the symptom is seen...
4. `medquad_ans_0000392-1` (score=0.0161) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disorders. These factors ...
5. `medquad_ans_0000392-5` (score=0.0159) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing Registry: Fryns syndr...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

Note: There appears to be a confusion in the name "Fryns Hofkens Fabry syndrome." Fabry disease is a well-documented condition, while "Fryns Hofkens Fabry syndrome" is not a recognized or standard medical term in the provided context. The symptoms listed are more consistent with those of Fabry disease or a related condition, and the term "Fryns Hofkens Fabry syndrome" may be a misstatement or error. Based on the provided context, the listed symptoms are associated with a condition that may be

**Citations detected**: `medquad_ans_0002456-1`
**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

## Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Top docs**:

1. `medquad_ans_0000009_2-8` (score=0.0262) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ongoing clinical tri...
2. `medquad_ans_0000019_3-7` (score=0.0256) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, taking part in a clini...
3. `medquad_ans_0000019_2-7` (score=0.0244) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000006_3-8` (score=0.0238) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
5. `medquad_ans_0000025_1-8` (score=0.0234) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for osteoporosis. The information discusses clinical trials for osteosarcoma, a type of bone cancer, not osteoporosis [medquad_ans_0000009_2-8]. Therefore, based on the given context, no research or clinical trials for osteoporosis are described.

**Citations detected**: `medquad_ans_0000009_2-8`
**Supported citations**: `medquad_ans_0000009_2-8`
**Unsupported citations**: _none_

## Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.0328) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. At worst, they're da...
2. `medquad_ans_0000286-1__dup2` (score=0.0278) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do. Some supplements c...
3. `medquad_ans_0000925-1__dup1` (score=0.0252) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about your baby's health, ca...
4. `medquad_ans_0000694-1__dup1` (score=0.0247) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of medical errors, you ca...
5. `medquad_ans_0000693-1__dup2` (score=0.0211) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protecting patients, and h...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, secret ingredients, or scientific breakthroughs for common health issues like weight loss, memory loss, sexual performance, and joint pain. They target individuals with serious conditions such as cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Health fraud can be dangerous, waste money, and prevent people from accessing effective treatments. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments [medquad_ans_0000430-1__dup1].

**Citations detected**: `medquad_ans_0000430-1__dup1`
**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

## Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Top docs**:

1. `medquad_ans_0000054-1__dup4` (score=0.0298) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your blood for a long time...
2. `medquad_ans_0000009-3__dup3` (score=0.0298) - There are several things you can do to lower your risk for cataract. They include - having regular eye exams - quitting smoking - wearing sunglasses - taking care of other health problems - maintaining a healthy weight - choosing a healt...
3. `medquad_ans_0000078-15` (score=0.0296) - You can do a lot to prevent or delay type 2 diabetes by making these lifestyle changes: - Reach and stay at a healthy weight. Try to reach your prepregnancy weight 6 to 12 months after your baby is born. Then, if you still weigh too much...
4. `medquad_ans_0000054-12__dup2` (score=0.0278) - You can take steps each day to prevent diabetes problems. Steps Healthy Eating - Follow the healthy eating plan that you and your doctor or dietitian have made. - Learn what to eat to keep your blood glucose levels under control. - Make ...
5. `medquad_ans_0000275-1__dup1` (score=0.0269) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adults. Your retina is t...

**Answer**:

The provided context does not include information about the symptoms of preventing diabetes problems or specifically about keeping eyes healthy in relation to diabetes prevention. While it mentions that diabetes can affect the eyes, leading to loss of vision and blindness, and recommends regular eye exams for early detection of cataracts and other eye problems, it does not describe symptoms associated with preventing diabetes-related eye issues [medquad_ans_0000054-1\_\_dup4][medquad_ans_0000009-3__dup3]. Therefore, based on the given context, the symptoms of preventing diabetes problems or keeping eyes healthy cannot be answered.

**Citations detected**: `medquad_ans_0000054-1__dup4`, `medquad_ans_0000009-3__dup3`
**Supported citations**: `medquad_ans_0000054-1__dup4`, `medquad_ans_0000009-3__dup3`
**Unsupported citations**: _none_

## Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000001-4` (score=0.0164) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000001-1` (score=0.0161) - Aarskog-Scott syndrome is a genetic disorder that affects the development of many parts of the body. This condition mainly affects males, although females may have mild features of the syndrome. People with Aarskog-Scott syndrome often h...
4. `medquad_ans_0000424-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000001-2` (score=0.0159) - Aarskog-Scott syndrome is believed to be a rare disorder; however, its prevalence is unknown because mildly affected people are often not diagnosed.

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Citations detected**: `medquad_ans_0000001-4`
**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

## Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Top docs**:

1. `medquad_ans_0001063-4` (score=0.0320) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
2. `medquad_ans_0006525-1` (score=0.0318) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that are smaller than no...
3. `medquad_ans_0001063-1` (score=0.0300) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anemia results when deve...
4. `medquad_ans_0001062-4` (score=0.0300) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
5. `medquad_ans_0001062-1` (score=0.0296) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linked sideroblastic anem...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. This disorder is caused by a mutation in the ALAS2 gene, which is located on the X chromosome and is inherited in an X-linked recessive pattern [medquad_ans_0006525-1]. In rare cases, mutations in both the HFE gene and the ALAS2 gene can result in a more severe form of the condition [medquad_ans_0006525-1].

**Citations detected**: `medquad_ans_0006525-1`
**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

## Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.0313) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifically the muscles of...
2. `medquad_ans_0003604-1` (score=0.0313) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around the hips and shoulde...
3. `medquad_ans_0003610-1` (score=0.0312) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdle muscular dystrophi...
4. `medquad_ans_0003601-1` (score=0.0306) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of years. Onset may occu...
5. `medquad_ans_0003608-2` (score=0.0292) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2I? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2I. If the information is available, the ta...

**Answer**:

Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs, primarily affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. The severity, age of onset, and features vary among the subtypes and may differ even within the same family. Signs and symptoms may appear at any age and generally worsen over time, though some cases remain mild. Early symptoms may include an unusual walking gait, difficulty running, and trouble rising from a squatting position due to weak thigh muscles. As the condition progresses, individuals may require wheelchair assistance, develop scapular winging, lordosis, scoliosis, joint stiffness, calf muscle hypertrophy, or cardiomyopathy in some forms. Breathing problems may occur due to weakened respiratory

**Citations detected**: `medquad_ans_0000602-1`
**Supported citations**: `medquad_ans_0000602-1`
**Unsupported citations**: _none_

## Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Top docs**:

1. `medquad_ans_0000343-1__dup1` (score=0.0315) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage to the optic nerve - ...
2. `medquad_ans_0000275-1__dup1` (score=0.0315) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adults. Your retina is t...
3. `medquad_ans_0000016-6` (score=0.0308) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
4. `medquad_ans_0000016-2__dup2` (score=0.0301) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a compreh...
5. `medquad_ans_0000016-4__dup2` (score=0.0286) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataracts, and glaucoma. Diabetic retinopathy is a complication of diabetes that occurs when high blood glucose levels damage the tiny blood vessels in the retina, the light-sensitive tissue at the back of the eye. It is a leading cause of blindness in American adults and may initially show no symptoms, though over time it can cause vision loss. Cataracts involve a clouding of the lens of the eye, and glaucoma occurs when pressure builds up in the eye, damaging the main nerve. These conditions can be detected early through regular eye exams, and treatment such as laser therapy, surgery, or eye drops may help prevent vision loss [medquad_ans_0000275-1\_\_dup1][medquad_ans_0000016-6].

**Citations detected**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000016-6`
**Supported citations**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000016-6`
**Unsupported citations**: _none_

## Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.0313) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0000133-1__dup3` (score=0.0310) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filters extra copper and...
3. `medquad_ans_0000133-13` (score=0.0309) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correctly and copper build...
4. `medquad_ans_0000973-1__dup1` (score=0.0308) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver releases extra copper i...
5. `medquad_ans_0006449-5` (score=0.0308) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, whi...

**Answer**:

Wilson disease is a rare genetic disorder that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filters extra copper and releases it into bile, a fluid that carries toxins and wastes out of the body through the gastrointestinal tract. In Wilson disease, the liver does not filter copper correctly, leading to copper buildup in the liver, brain, eyes, and other organs. This buildup can cause life-threatening organ damage. Wilson disease is caused by an inherited autosomal recessive mutation in the ATP7B gene, meaning a child must inherit the mutation from both parents to have an increased likelihood of developing the disease. The signs and symptoms vary depending on the organs affected and may include liver-related, central nervous system-related, and mental health-related issues. Treatment involves a lifelong effort to reduce and control copper levels, including medications, dietary changes (such as avoiding high

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Top docs**:

1. `medquad_ans_0001504-1` (score=0.0325) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the information is available...
2. `medquad_ans_0001503-2` (score=0.0318) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to absorb too much iron, w...
3. `medquad_ans_0002778-1` (score=0.0287) - What are the signs and symptoms of Hemolytic anemia lethal congenital nonspherocytic with genital and other abnormalities? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemolytic anemia lethal congeni...
4. `medquad_ans_0006525-2` (score=0.0254) - What are the signs and symptoms of X-linked sideroblastic anemia? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked sideroblastic anemia. If the information is available, the table below includes...
5. `medquad_ans_0006061-2` (score=0.0236) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrome. If the informatio...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen, and iron overload, which can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur. Blood transfusions may be necessary during pregnancy or other special circumstances like anemic crisis, major surgery, or infections [medquad_ans_0001503-2].

**Citations detected**: `medquad_ans_0001503-2`
**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

## Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000038-3__dup2` (score=0.0271) - Kidney Disease Kidney disease is often called a "silent" disease, because most people have no symptoms with early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for symptoms!...
2. `medquad_ans_0000136-12` (score=0.0265) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need to know about Kidney...
3. `medquad_ans_0000038-8__dup1` (score=0.0263) - Many areas of Research Researchers are working at every stage of kidney disease to improve diagnosis and treatment, including - trying to find a better way to identify who is at greatest risk for rapidly progressing kidney disease - tryi...
4. `medquad_ans_0000138-11` (score=0.0260) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - Treatment Methods for Kidney Fai...
5. `medquad_ans_0000179-14` (score=0.0257) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need to know about Kidney...

**Answer**:

Kidney disease is often called a "silent" disease because most people have no symptoms in the early stages, and individuals might feel fine until their kidneys have almost stopped working. It is characterized by damaged kidneys that filter less than 15% of the blood filtered by healthy kidneys, leading to a condition known as uremia, where waste products build up in the blood. Kidney disease can also result in hormonal imbalances and disruptions in mineral balance in the blood, causing complications such as itching, sleep problems, restless legs, weak bones, joint problems, and depression. Diagnosis is typically made through blood and urine tests, which assess kidney function using the glomerular filtration rate (GFR). Key risk factors include diabetes, high blood pressure, heart disease, and a family history of kidney failure [medquad_ans_0000038-3__dup2].

**Citations detected**: `medquad_ans_0000038-3__dup2`
**Supported citations**: `medquad_ans_0000038-3__dup2`
**Unsupported citations**: _none_

## Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.0325) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But doctors also conside...
2. `medquad_ans_0000036_1-6` (score=0.0320) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopharmaceutical therapy ...
3. `medquad_ans_0000036_1-7` (score=0.0294) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000750-1__dup1` (score=0.0286) - The prostate is the gland below a man's bladder that produces fluid for semen. Prostate cancer is common among older men. It is rare in men younger than 40. Risk factors for developing prostate cancer include being over 65 years of age, ...
5. `medquad_ans_0000058-6__dup1` (score=0.0282) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But d...

**Answer**:

The treatments for prostate cancer include watchful waiting or active surveillance, surgery, radiation therapy and radiopharmaceutical therapy, hormone therapy, chemotherapy, biologic therapy, and bisphosphonate therapy. Treatments for bone pain caused by bone metastases or hormone therapy are also available. Additionally, new types of treatments such as cryosurgery, high-intensity focused ultrasound therapy, and proton beam radiation therapy are being used. Patients may also consider participating in clinical trials, which are research studies aimed at improving current treatments or obtaining information on new treatments [medquad_ans_0000036_1-6].

**Citations detected**: `medquad_ans_0000036_1-6`
**Supported citations**: `medquad_ans_0000036_1-6`
**Unsupported citations**: _none_

## Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.0311) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when you need medical care....
2. `medquad_ans_0000005-19` (score=0.0308) - Most insurance plans, including health maintenance organizations (HMOs), will cover treatment for anxiety disorders. Check with your insurance company and find out. If you dont have insurance, the Health and Human Services division of yo...
3. `medquad_ans_0000362-1__dup2` (score=0.0308) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance might be available. Cer...
4. `medquad_ans_0000196-1__dup3` (score=0.0293) - We all want high-quality health care, but it's hard to know how to choose. There are many things to consider, including - What your insurance covers - Whether a health care provider or service is accredited - The location of a service - ...
5. `medquad_ans_0000586-1__dup2` (score=0.0283) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state has its own rules. You...

**Answer**:

Yes, there is information about health insurance. Health insurance helps protect individuals from high medical care costs by being a contract between the individual and an insurance company, where the company agrees to pay part of the expenses when medical care is needed [medquad_ans_0000431-1__dup1]. Many people in the United States obtain health insurance through their employers, and employer-sponsored plans are often managed care plans that contract with healthcare providers to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Individuals can also purchase health insurance independently, and certain groups may qualify for government health insurance such as Medicare and Medicaid [medquad_ans_0000431-1__dup1]. Additionally, the Affordable Care Act has expanded health insurance coverage for many people in the U.S. [medquad_ans_0000431-1__dup1].

**Citations detected**: `medquad_ans_0000431-1__dup1`
**Supported citations**: `medquad_ans_0000431-1__dup1`
**Unsupported citations**: _none_

## Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.0328) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...
2. `medquad_ans_0003860-4` (score=0.0315) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome, one of the two sex...
3. `medquad_ans_0000635-3` (score=0.0313) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
4. `medquad_ans_0003860-5` (score=0.0311) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the diagnosis.
5. `medquad_ans_0000635-4` (score=0.0308) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), a duplication of ...

**Answer**:

MECP2 duplication syndrome is a genetic disorder caused by an extra copy (duplication) of the MECP2 gene in each cell, typically located on the long (q) arm of the X chromosome. This duplication results in the production of excess MECP2 protein, which disrupts the normal regulation of other genes in the brain, leading to irregular brain activity and the signs and symptoms of the condition. The syndrome is inherited in an X-linked manner; in males, a duplication of the MECP2 gene is sufficient to cause the disorder, while in females, a duplication of one copy of the gene often does not lead to symptoms due to random X-inactivation, which may preferentially inactivate the chromosome with the duplicated gene (skewed X-inactivation). Most cases are inherited from a mother who has no signs or symptoms, though some cases occur de novo due to a random genetic event during egg or sperm formation or early fetal development [medquad

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Top docs**:

1. `medquad_ans_0000017_1-1` (score=0.0243) - Key Points - Intraocular melanoma is a disease in which malignant (cancer) cells form in the tissues of the eye. - Being older and having fair skin may increase the risk of intraocular melanoma. - Signs of intraocular melanoma include bl...
2. `medquad_ans_0000594-1__dup2` (score=0.0234) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as a new mole. It may b...
3. `medquad_ans_0000028_2-6` (score=0.0227) - Key Points - After melanoma has been diagnosed, tests are done to find out if cancer cells have spread within the skin or to other parts of the body. - There are three ways that cancer spreads in the body. - Cancer may spread from where ...
4. `medquad_ans_0000063-3__dup3` (score=0.0221) - Early Detection is Important When skin cancer is found early, it is more likely to be treated successfully. Therefore, it is important to know how to recognize the signs of skin cancer in order to improve the chances of early diagnosis. ...
5. `medquad_ans_0000028_2-1` (score=0.0164) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the skin. - Unusual mole...

**Answer**:

The provided context does not contain information about the outlook (prognosis) for melanoma. While it describes intraocular melanoma in detail, including its location and risk factors, it does not address prognosis or survival rates for melanoma in general. Therefore, based on the given context, the outlook for melanoma cannot be determined.

[doc_id]medquad_ans_0000594-1\_\_dup2[/doc_id]

**Citations detected**: `doc_id`, `/doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`, `/doc_id`

## Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Top docs**:

1. `medquad_ans_0000132-2` (score=0.0164) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000132-1` (score=0.0161) - Boomerang dysplasia is a disorder that affects the development of bones throughout the body. Affected individuals are born with inward- and upward-turning feet (clubfeet) and dislocations of the hips, knees, and elbows. Bones in the spin...
4. `medquad_ans_0000298-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000132-5` (score=0.0159) - These resources address the diagnosis or management of boomerang dysplasia: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Boomerang dysplasia These resources from MedlinePlus offer information about the d...

**Answer**:

Yes, boomerang dysplasia is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs and symptoms of the condition [medquad_ans_0000785-4]. This information is also supported by [medquad_ans_0000298-4].

**Citations detected**: `medquad_ans_0000785-4`, `medquad_ans_0000298-4`
**Supported citations**: `medquad_ans_0000785-4`, `medquad_ans_0000298-4`
**Unsupported citations**: _none_

## Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Top docs**:

1. `medquad_ans_0002838-1` (score=0.0323) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called osteochondromas). T...
2. `medquad_ans_0000466-3` (score=0.0315) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exostosin proteins bind to...
3. `medquad_ans_0002838-4` (score=0.0302) - Is genetic testing available for hereditary multiple osteochondromas? GeneTests lists the names of laboratories that are performing genetic testing for hereditary multiple osteochondromas. To view the contact information for the clinical...
4. `medquad_ans_0002838-3` (score=0.0299) - How is hereditary multiple osteochondromas inherited? HMO is caused by mutations in the EXT1 and EXT2 genes. It is inherited in an autosomal dominant pattern, which means that one copy of the altered gene in each cell is sufficient to ca...
5. `medquad_ans_0000466-1` (score=0.0292) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary greatly among affect...

**Answer**:

The genetic changes related to hereditary multiple osteochondromas (HMO) are mutations in the EXT1 and EXT2 genes. These mutations lead to the production of defective exostosin-1 or exostosin-2 proteins, which form a complex in the Golgi apparatus responsible for modifying heparan sulfate. When there is a mutation in exostosin-1 or exostosin-2, heparan sulfate cannot be processed correctly and becomes nonfunctional. Mutations in the EXT1 gene result in hereditary multiple osteochondromas type 1, while mutations in the EXT2 gene cause type 2. Mutations in the EXT1 gene are likely responsible for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals with HMO have no identifiable mutation in either EXT1 or EXT2 [medquad_ans

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Top docs**:

1. `medquad_ans_0000746-5` (score=0.0164) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosis and management of ...
2. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000746-1` (score=0.0161) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abnormalities associate...
4. `medquad_ans_0000989-2` (score=0.0161) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0004510-1` (score=0.0159) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip that is divided by a cl...

**Answer**:

Oculofaciocardiodental syndrome affects about 1 per million people worldwide, and it is estimated to affect 1 in 50,000 people [medquad_ans_0000359-2][medquad_ans_0000989-2].

**Citations detected**: `medquad_ans_0000359-2`, `medquad_ans_0000989-2`
**Supported citations**: `medquad_ans_0000359-2`, `medquad_ans_0000989-2`
**Unsupported citations**: _none_

## Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Top docs**:

1. `medquad_ans_0001350-5` (score=0.0325) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention, hyperammonemia and...
2. `medquad_ans_0000189-1` (score=0.0315) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in diffe...
3. `medquad_ans_0000189-2` (score=0.0313) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100,000 to 230,000 indi...
4. `medquad_ans_0001350-4` (score=0.0312) - How is citrullinemia type I inherited? Citrullinemia type I is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive conditi...
5. `medquad_ans_0001350-2` (score=0.0308) - What are the signs and symptoms of Citrullinemia type I? Citrullinemia type I presents as a clinical spectrum that includes an acute neonatal form, a milder late-onset form, a form without symptoms and/or hyperammonemia, and a form in wh...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in different genes. Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life. Affected infants typically appear normal at birth, but as ammonia builds up in the body they experience a progressive lack of energy (lethargy), poor feeding, vomiting, seizures, and loss of consciousness. These medical problems are life-threatening in many cases. Less commonly, a milder form of type I citrullinemia can develop later in childhood or adulthood, associated with intense headaches, partial loss of vision, problems with balance and muscle coordination (ataxia), and lethargy. Type II citrullinemia chiefly affects the nervous system, causing

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.0323) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius...
2. `medquad_ans_0000720-1__dup1` (score=0.0315) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the entire surface of the br...
3. `medquad_ans_0000807-1` (score=0.0309) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops too many folds, and...
4. `medquad_ans_0000719-1__dup1` (score=0.0303) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides of the brain (bilate...
5. `medquad_ans_0000722-4__dup1` (score=0.0301) - Is bilateral perisylvian polymicrogyria inherited? In most cases, bilateral perisylvian polymicrogyria (BPP) occurs sporadically in people with no family history of the condition. Rarely, more than one family member may be affected by BP...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form that affects only the frontal lobes of the brain, without involving the areas located behind the Sylvius fissure or the Rolando sulcus. Signs and symptoms include delayed motor and language milestones, spastic hemiparesis or quadriparesis, and mild to moderate intellectual disability. Seizures may also be present. The cause of this form is still not known, unlike the frontoparietal form, which is associated with mutations in the GPR56 gene [medquad_ans_0000718-1__dup1]. Treatment is based on the individual's specific signs and symptoms [medquad_ans_0000718-1__dup1].

**Citations detected**: `medquad_ans_0000718-1__dup1`
**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

## Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Top docs**:

1. `medquad_ans_0000944-5` (score=0.0306) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic Testing Registry: Sp...
2. `medquad_ans_0005860-4` (score=0.0281) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respiratory infections (p...
3. `medquad_ans_0000944-3` (score=0.0164) - Mutations in at least four genes are known to cause spondylocostal dysostosis: Mutations in the DLL3 gene cause spondylocostal dysostosis type 1; mutations in the MESP2 gene cause spondylocostal dysostosis type 2; mutations in the LFNG g...
4. `medquad_ans_0000949-5` (score=0.0164) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from MedlinePlus offer informa...
5. `medquad_ans_0005821-1` (score=0.0161) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an abnormal side-to-si...

**Answer**:

Treatment for spondylocostal dysostosis includes intensive medical care, such as treatment of respiratory infections, bone surgery, and orthopedic treatment. Infants with the condition often have difficulty breathing due to small, malformed chests, making them prone to repeated respiratory infections and potentially leading to life-threatening complications as the chest cannot accommodate growing lungs [medquad_ans_0005860-4].

**Citations detected**: `medquad_ans_0005860-4`
**Supported citations**: `medquad_ans_0005860-4`
**Unsupported citations**: _none_

## Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.0323) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often the symptom is seen...
2. `medquad_ans_0000786-1` (score=0.0305) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (hypoventilation). The...
3. `medquad_ans_0000786-5` (score=0.0161) - These resources address the diagnosis or management of Perry syndrome: - Gene Review: Gene Review: Perry Syndrome - Genetic Testing Registry: Perry syndrome - MedlinePlus Encyclopedia: Major Depression - MedlinePlus Encyclopedia: Primary...
4. `medquad_ans_0006409-2` (score=0.0161) - What are the signs and symptoms of Weaver syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weaver syndrome. If the information is available, the table below includes how often the symptom is se...
5. `medquad_ans_0003679-1` (score=0.0159) - What are the signs and symptoms of Lowry Wood syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lowry Wood syndrome. If the information is available, the table below includes how often the sympt...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function (such as parkinsonism, bradykinesia, rigidity, tremor, and mask-like facies), respiratory insufficiency (including hypoventilation and central hypoventilation), sleep disturbance, weight loss, developmental regression, hallucinations, hypotension, apathy, inappropriate behavior, insomnia, dysarthria, and Parkinsonism [medquad_ans_0004825-1]. Psychiatric changes such as depression, apathy, withdrawal from social interactions, and suicidal thoughts are also common, and these often appear early in the disease [medquad_ans_0000786-1]. The condition typically presents in a person's forties or fifties, with parkinsonism and psychiatric changes being the earliest features [medquad_ans_0000786-1].

**Citations detected**: `medquad_ans_0004825-1`, `medquad_ans_0000786-1`
**Supported citations**: `medquad_ans_0004825-1`, `medquad_ans_0000786-1`
**Unsupported citations**: _none_

## Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Top docs**:

1. `medquad_ans_0001028-2` (score=0.0323) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people in the U.S. who are...
2. `medquad_ans_0006398-2` (score=0.0307) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0006399-1` (score=0.0300) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how often the symptom i...
4. `medquad_ans_0006398-1` (score=0.0282) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increased risk for blood cl...
5. `medquad_ans_0000108-2__dup1` (score=0.0273) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...

**Answer**:

The most common sign of warfarin sensitivity is excessive internal bleeding, which often occurs when individuals with warfarin sensitivity are given too much of the medication [medquad_ans_0001028-2]. The Human Phenotype Ontology lists additional signs and symptoms associated with warfarin sensitivity, including abnormalities of blood and blood-forming tissues, abnormalities of metabolism/homeostasis, and autosomal dominant inheritance. However, specific frequencies for these symptoms are not provided in the context [medquad_ans_0006398-2]. Note that the symptoms listed in the context for "Warfarin syndrome" may differ from those of warfarin sensitivity, and no direct link is established between the two conditions in the provided information.

**Citations detected**: `medquad_ans_0001028-2`, `medquad_ans_0006398-2`
**Supported citations**: `medquad_ans_0001028-2`, `medquad_ans_0006398-2`
**Unsupported citations**: _none_

## Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Top docs**:

1. `medquad_ans_0004262-3` (score=0.0320) - What causes Myhre syndrome? Myhre syndrome is caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus. This signaling path...
2. `medquad_ans_0000701-1` (score=0.0315) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most have intellectual d...
3. `medquad_ans_0004262-1` (score=0.0313) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement ...
4. `medquad_ans_0004262-5` (score=0.0309) - How might Myhre syndrome be treated? Treatment of this condition is symptomatic and supportive. Children with Myhre syndrome may require management by a team of specialists, including pediatricians, speech pathologists, orthopedists (bon...
5. `medquad_ans_0000701-2` (score=0.0308) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.

**Answer**:

Myhre syndrome is a condition with features affecting many systems and functions of the body. It is caused by mutations in the SMAD4 gene, which disrupts the transforming growth factor beta (TGF-) signaling pathway by impairing the proper binding of the SMAD4 protein to other proteins or resulting in an abnormally stable SMAD4 protein that remains active longer [medquad_ans_0004262-3]. This abnormal signaling affects development across multiple body systems and leads to a range of signs and symptoms. These include delayed development of language and motor skills, intellectual disability ranging from mild to moderate, behavioral issues such as autism-related disorders, hearing loss (sensorineural, conductive, or mixed), reduced growth with low birth weight and short stature, skin stiffness and a muscular appearance, skeletal abnormalities like thickened skull bones, platyspondyly, broad ribs, hypoplastic iliac wings, and brachyd

**Citations detected**: `medquad_ans_0004262-3`
**Supported citations**: `medquad_ans_0004262-3`
**Unsupported citations**: _none_

## Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.0311) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below includes how often t...
2. `medquad_ans_0004193-1` (score=0.0275) - What are the signs and symptoms of Multiple epiphyseal dysplasia 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple epiphyseal dysplasia 4. If the information is available, the table below incl...
3. `medquad_ans_0005848-1` (score=0.0260) - What are the signs and symptoms of Spondylometaphyseal dysplasia Algerian type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia Algerian type. If the information is availab...
4. `medquad_ans_0005849-1` (score=0.0197) - What are the signs and symptoms of Spondylometaphyseal dysplasia corner fracture type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia corner fracture type. If the informat...
5. `medquad_ans_0006057-4` (score=0.0164) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of thanatophoric dysplasia...

**Answer**:

The symptoms of Thanatophoric dysplasia include abnormality of the femur, abnormality of the metaphyses, abnormality of the sacroiliac joint, aplasia/hypoplasia of the lungs, bowing of the long bones, brachydactyly syndrome, cognitive impairment, cutis laxa, depressed nasal bridge, increased nuchal translucency, macrocephaly, malar flattening, micromelia, muscular hypotonia, narrow chest, platyspondyly, respiratory insufficiency, short stature, short thorax, skeletal dysplasia, small face, split hand, abnormality of neuronal migration, frontal bossing, hearing impairment, intrauterine growth retardation, kyphosis, polyhydramnios, proptosis, ventriculomegaly, abnormality of the kidney, acanthosis nigricans, atria septal defect,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.0320) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually more severely affected ...
2. `medquad_ans_0000381-1` (score=0.0260) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females. Affected individua...
3. `medquad_ans_0000381-3` (score=0.0164) - Mutations in the FMR1 gene cause fragile X syndrome. The FMR1 gene provides instructions for making a protein called FMRP. This protein helps regulate the production of other proteins and plays a role in the development of synapses, whic...
4. `medquad_ans_0002417-3` (score=0.0161) - What causes fragile X syndrome? Mutations (changes) in the FMR1 gene cause fragile X syndrome (FXS). This gene carries instructions to make a protein called the fragile X mental retardation 1 protein. The FMR1 gene contains a section of ...
5. `medquad_ans_0003942-1` (score=0.0161) - What are the signs and symptoms of Mental retardation X-linked, South African type? The Human Phenotype Ontology provides the following list of signs and symptoms for Mental retardation X-linked, South African type. If the information is...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Citations detected**: `medquad_ans_0002417-2`
**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

## Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.0318) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000056-1__dup1` (score=0.0315) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the flow of blood to the b...
3. `medquad_ans_0000889-1__dup1` (score=0.0315) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow of blood to the brai...
4. `medquad_ans_0000889-5__dup1` (score=0.0315) - How might CADASIL be treated? There is currently no treatment for CADASIL that is proven to be effective. While antiplatelet treatment is often used, it is also not proven to be useful. Migraine should be treated both symptomatically and...
5. `medquad_ans_0000889-2__dup1` (score=0.0306) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average age at onset for str...

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease caused by a mutation in the NOTCH3 gene, which leads to thickening of the blood vessel walls and blocks blood flow to the brain, primarily affecting small blood vessels in the white matter of the brain [medquad_ans_0000056-1__dup1]. It is characterized by migraine headaches, multiple strokes, cognitive deterioration, seizures, vision problems, psychiatric symptoms such as severe depression and behavioral changes, and an increased risk of heart attack [medquad_ans_0000056-1__dup1]. The disease follows an autosomal dominant inheritance pattern, meaning one parent can pass on the defective gene, and most individuals have a family history of the disorder [medquad_ans_00008

**Citations detected**: `medquad_ans_0000056-1__dup1`
**Supported citations**: `medquad_ans_0000056-1__dup1`
**Unsupported citations**: _none_

## Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Top docs**:

1. `medquad_ans_0000688-1` (score=0.0325) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cel...
2. `medquad_ans_0000688-5` (score=0.0310) - These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Regist...
3. `medquad_ans_0000688-2` (score=0.0309) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followed by FMTC. Type 2B i...
4. `medquad_ans_0004187-2` (score=0.0294) - What are the signs and symptoms of Multiple endocrine neoplasia type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple endocrine neoplasia type 2A. If the information is available, the table ...
5. `medquad_ans_0004187-3` (score=0.0290) - How is multiple endocrine neoplasia type 2A inherited? Multiple endocrine neoplasia type 2A (MEN 2A) is inherited in an autosomal dominant pattern. A person with MEN 2A often inherits the altered RET gene from one parent with the condition.

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with multiple endocrine neoplasia. Type 1 frequently involves tumors of the

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.0303) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs)....
2. `medquad_ans_0004155-1` (score=0.0285) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by mutations in the IDUA...
3. `medquad_ans_0004155-4` (score=0.0280) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
4. `medquad_ans_0004155-2` (score=0.0274) - What are the signs and symptoms of Mucopolysaccharidosis type I? The signs and symptoms of MPS I are not present at birth, but they begin to appear during childhood. People with severe MPS I develop the features of this condition earlier...
5. `medquad_ans_0000678-1` (score=0.0272) - Mucopolysaccharidosis type I (MPS I) is a condition that affects many parts of the body. This disorder was once divided into three separate syndromes: Hurler syndrome (MPS I-H), Hurler-Scheie syndrome (MPS I-H/S), and Scheie syndrome (MP...

**Answer**:

Mutations in the IDUA gene cause Mucopolysaccharidosis type I (MPS I). The IDUA gene provides instructions for producing an enzyme involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme, leading to the accumulation of GAGs within cells, specifically inside the lysosomes. This accumulation causes lysosomes to increase in size and results in the enlargement of many tissues and organs [medquad_ans_0004155-3]. [medquad_ans_0004155-1]

**Citations detected**: `medquad_ans_0004155-3`, `medquad_ans_0004155-1`
**Supported citations**: `medquad_ans_0004155-3`, `medquad_ans_0004155-1`
**Unsupported citations**: _none_

## Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.0323) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called folliculin. In the lungs, f...
2. `medquad_ans_0005065-1` (score=0.0320) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the pleural space (the space between the lungs and the chest cavity) that can result in the partial or complete collapse of a lung. It is called primary because it oc...
3. `medquad_ans_0000825-4` (score=0.0318) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, a person inherits th...
4. `medquad_ans_0000825-1` (score=0.0311) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the space between the lungs and the chest cavity (called the pleural space) that can result in the partial or complete collapse of a lung. This type of pneumothorax i...
5. `medquad_ans_0005065-2` (score=0.0303) - What are the signs and symptoms of Primary spontaneous pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Primary spontaneous pneumothorax. If the information is available, the table below in...

**Answer**:

Genetic changes related to primary spontaneous pneumothorax include mutations in the FLCN gene. These mutations can cause primary spontaneous pneumothorax, although they appear to be a very rare cause of the condition. The FLCN gene provides instructions for making a protein called folliculin, which is found in lung connective tissue cells and in cells that line the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. One theory is that altered folliculin protein may trigger inflammation in lung tissue, leading to tissue damage and the formation of blebs, which can rupture and cause pneumothorax [medquad_ans_0000825-3]. In rare cases, primary spontaneous pneumothorax is caused by FLCN gene mutations [medquad_ans_005065-1].

**Citations detected**: `medquad_ans_0000825-3`, `medquad_ans_005065-1`
**Supported citations**: `medquad_ans_0000825-3`
**Unsupported citations**: `medquad_ans_005065-1`

## Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Top docs**:

1. `medquad_ans_0000066-1__dup4` (score=0.0320) - Stroke -- A Serious Event A stroke is serious, just like a heart attack. Each year in the United States, approximately 795,000 people have a stroke. About 610,000 of these are first or new strokes. On average, one American dies from stro...
2. `medquad_ans_0000066-13` (score=0.0320) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blood vessel in the bra...
3. `medquad_ans_0000066-5__dup2` (score=0.0308) - Stroke is preventable and treatable. A better understanding of the causes of stroke has helped people make lifestyle changes that have cut the stroke death rate nearly in half in the last two decades. Preventing Stroke While family histo...
4. `medquad_ans_0000527-1__dup2` (score=0.0301) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This keeps blood from flowin...
5. `medquad_ans_0000857-1__dup2` (score=0.0301) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by a blood clot that bl...

**Answer**:

A stroke is sometimes called a "brain attack." Most often, stroke occurs when blood flow to the brain stops because it is blocked by a clot. When this happens, the brain cells in the immediate area begin to die. Some brain cells die because they stop getting the oxygen and nutrients they need to function. Other brain cells die because they are damaged by sudden bleeding into or around the brain. The brain cells that don't die immediately remain at risk for death. These cells can linger in a compromised or weakened state for several hours. With timely treatment, these cells can be saved. New treatments are available that greatly reduce the damage caused by a stroke. But you need to arrive at the hospital as soon as possible after symptoms start to prevent disability and to greatly improve your chances for recovery. Knowing stroke symptoms, calling 911 immediately, and getting to a hospital as quickly as possible are critical [medquad_ans_0000066

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Top docs**:

1. `medquad_ans_0006276-5` (score=0.0323) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. When the individual ...
2. `medquad_ans_0006276-3` (score=0.0320) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been suggested that the co...
3. `medquad_ans_0006276-2` (score=0.0318) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of hair remains normal...
4. `medquad_ans_0006276-4` (score=0.0313) - Is uncombable hair syndrome inherited? Uncombable hair syndrome (UHS) is thought to be inherited in an autosomal dominant manner with reduced penetrance. Autosomal dominant means that having a change (mutation) in only one copy of the re...
5. `medquad_ans_0006276-1` (score=0.0310) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed flat. It may first b...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been suggested that the condition may result from premature keratinization (development of keratin) of the inner root sheath, which forms the channel for the growing hair. The inner root sheath conforms in configuration to the abnormal outline of the hair shaft. It thus forms an irregular, rigid tube that then alters the shape of the emerging hair. While it is assumed that the condition is autosomal dominant and thus due to changes (mutations) in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Citations detected**: `medquad_ans_0006276-3`
**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

## Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000887-4` (score=0.0308) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected families, the condition ap...
2. `medquad_ans_0000887-3` (score=0.0164) - The genetic causes of Russell-Silver syndrome are complex. The disorder often results from the abnormal regulation of certain genes that control growth. Research has focused on genes located in particular regions of chromosome 7 and chro...
3. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0005440-4` (score=0.0161) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is inherited. In some ...
5. `medquad_ans_0000887-1` (score=0.0159) - Russell-Silver syndrome is a growth disorder characterized by slow growth before and after birth. Babies with this condition have a low birth weight and often fail to grow and gain weight at the expected rate (failure to thrive). Head gr...

**Answer**:

Russell-Silver syndrome can be inherited, but most cases are sporadic. It may follow an autosomal dominant or autosomal recessive pattern of inheritance in some families. In autosomal dominant inheritance, one copy of a genetic change is sufficient to cause the disorder, while in autosomal recessive inheritance, both copies of the gene are altered. Additionally, the syndrome is associated with abnormalities in genomic imprinting, particularly involving genes like H19 and IGF2 on chromosome 11, and in some cases, with maternal uniparental disomy of chromosome 7 [medquad_ans_0000887-4, medquad_ans_0000887-3]. Therefore, while not all cases are inherited, the disorder can be passed down through families in specific genetic patterns.

**Citations detected**: `medquad_ans_0000887-4, medquad_ans_0000887-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000887-4, medquad_ans_0000887-3`

## Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.0328) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weigh...
2. `medquad_ans_0000095-7` (score=0.0315) - Successful weight-loss treatments include setting goals and making lifestyle changes, such as eating fewer calories and being physically active. Medicines and weight-loss surgery also are options for some people if lifestyle changes aren...
3. `medquad_ans_0000022-9` (score=0.0288) - You can do a lot to reduce your risk of getting type 2 diabetes. Being more physically active, reducing fat and calorie intake, and losing a little weight can help you lower your chances of developing type 2 diabetes. Taking the diabetes...
4. `medquad_ans_0000095-1__dup2` (score=0.0274) - Espaol The terms "overweight" and "obesity" refer to body weight thats greater than what is considered healthy for a certain height. The most useful measure of overweight and obesity is body mass index (BMI). BMI is calculated from your ...
5. `medquad_ans_0000078-15` (score=0.0246) - You can do a lot to prevent or delay type 2 diabetes by making these lifestyle changes: - Reach and stay at a healthy weight. Try to reach your prepregnancy weight 6 to 12 months after your baby is born. Then, if you still weigh too much...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Citations detected**: `medquad_ans_0000969-1__dup2`
**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

## Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.0323) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Saethre-Chotzen syndrom...
2. `medquad_ans_0000889-3` (score=0.0164) - Mutations in the TWIST1 gene cause Saethre-Chotzen syndrome. The TWIST1 gene provides instructions for making a protein that plays an important role in early development. This protein is a transcription factor, which means that it attach...
3. `medquad_ans_0000889-1` (score=0.0161) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and face. Most ...
4. `medquad_ans_0000172-2__dup1` (score=0.0161) - Treatment is symptomatic and supportive, and may include drug therapy.
5. `medquad_ans_0000583-5` (score=0.0159) - These resources address the diagnosis or management of laryngo-onycho-cutaneous syndrome: - Genetic Testing Registry: Laryngoonychocutaneous syndrome These resources from MedlinePlus offer information about the diagnosis and management o...

**Answer**:

The treatments for Saethre-Chotzen syndrome include surgical intervention, genetic counseling, and rehabilitation. Surgical management is often necessary to address craniosynostosis, which involves the premature fusion of skull bones, and to correct abnormalities in head and face shape. Additionally, rehabilitation and other supportive care may be provided to manage associated developmental and physical challenges. These approaches are part of the broader diagnostic and management strategies outlined in resources such as MedlinePlus, which covers surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000889-5].

**Citations detected**: `medquad_ans_0000889-5`
**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

## Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000512-1` (score=0.0323) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effectively. The disease u...
2. `medquad_ans_0000758-1__dup2` (score=0.0315) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...
3. `medquad_ans_0000512-3` (score=0.0304) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of developing idiopathic pul...
4. `medquad_ans_0000078-1__dup2` (score=0.0301) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens, your lungs can't p...
5. `medquad_ans_0000078-5__dup1` (score=0.0295) - The signs and symptoms of idiopathic pulmonary fibrosis (IPF) develop over time. They may not even begin to appear until the disease has done serious damage to your lungs. Once they occur, they're likely to get worse over time. The most ...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making it difficult for the lungs to transport oxygen into the bloodstream effectively [medquad_ans_0000512-1]. It typically affects individuals between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and clubbing (widened and rounded tips of the fingers and toes) [medquad_ans_0000512-1]. In most cases, the cause cannot be identified, and this is referred to as idiopathic pulmonary fibrosis [medquad_ans_0000758-1__dup2]. The disease progresses over time, leading to reduced lung function and potentially serious complications such as lung cancer, pulmonary emboli, pneumonia,

**Citations detected**: `medquad_ans_0000512-1`, `medquad_ans_0000758-1__dup2`
**Supported citations**: `medquad_ans_0000512-1`, `medquad_ans_0000758-1__dup2`
**Unsupported citations**: _none_

## Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=0.0323) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recovers, although there i...
2. `medquad_ans_0000213-4__dup1` (score=0.0309) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
3. `medquad_ans_0000006_4-7` (score=0.0288) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000024_3-8` (score=0.0251) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
5. `medquad_ans_0000024_4-7` (score=0.0247) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

Research on neurological disorders such as neuroleptic malignant syndrome is supported by the NINDS, with a focus on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. However, the provided context does not specify any ongoing clinical trials specifically for neuroleptic malignant syndrome. Information about clinical trials is available from the NCI website, but no details on trials related to neuroleptic malignant syndrome are included in the context [medquad_ans_0000006_4-7].

**Citations detected**: `medquad_ans_0000213-4__dup1`, `medquad_ans_0000006_4-7`
**Supported citations**: `medquad_ans_0000213-4__dup1`, `medquad_ans_0000006_4-7`
**Unsupported citations**: _none_

## Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Top docs**:

1. `medquad_ans_0000221-8` (score=0.0320) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other health problems. The d...
2. `medquad_ans_0000203-11` (score=0.0315) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary tract. - Certain fo...
3. `medquad_ans_0000204-11` (score=0.0292) - - A kidney stone is a solid piece of material that forms in a kidney when some substances that are normally found in the urine become highly concentrated. - Kidney stones occur in infants, children, and teenagers from all races and ethni...
4. `medquad_ans_0000221-12` (score=0.0292) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower levels. - Kidney ston...
5. `medquad_ans_0000221-9` (score=0.0273) - The treatment for kidney stones usually depends on their size and what they are made of. Kidney stones may be treated by your regular doctor or by a urologista doctor who specializes in the urinary tract. You may need treatment if you ha...

**Answer**:

To diagnose kidney stones in adults, a doctor will perform a physical exam and take a medical history, including questions about family history, diet, digestive problems, and other health issues. The doctor may also perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infections or substances in the urine that contribute to stone formation. Blood tests can identify underlying conditions that may lead to kidney stones. Imaging tests are used to locate the stones and may also reveal issues that caused the stone to form [medquad_ans_0000221-8].

**Citations detected**: `medquad_ans_0000221-8`
**Supported citations**: `medquad_ans_0000221-8`
**Unsupported citations**: _none_

## Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.0325) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of bladder cancer can a...
2. `medquad_ans_0000041_1-5` (score=0.0320) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. - Cancer may spread f...
3. `medquad_ans_0000041_1-2` (score=0.0305) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. Check with your docto...
4. `medquad_ans_0000041_1-6` (score=0.0294) - Key Points - There are different types of treatment for patients with urethral cancer. - Four types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Active surveillance - New types of treatment are being tes...
5. `medquad_ans_0000213-3__dup2` (score=0.0270) - Urinary retention can result from - obstruction of the urethra - nerve problems - medications - weakened bladder muscles Obstruction of the Urethra Obstruction of the urethra causes urinary retention by blocking the normal urine flow out...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Top docs**:

1. `medquad_ans_0004725-1` (score=0.0164) - Paramyotonia congenita is an inherited condition that affects muscles used for movement (skeletal muscles), mainly in the face, neck, arms, and hands. Symptoms begin in infancy or early childhood and include episodes of sustained muscle ...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000771-1` (score=0.0161) - Paramyotonia congenita is a disorder that affects muscles used for movement (skeletal muscles). Beginning in infancy or early childhood, people with this condition experience bouts of sustained muscle tensing (myotonia) that prevent musc...
4. `medquad_ans_0001011-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000771-2` (score=0.0159) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.

**Answer**:

Yes, paramyotonia congenita is inherited in an autosomal dominant pattern. [medquad_ans_0004725-1]

**Citations detected**: `medquad_ans_0004725-1`
**Supported citations**: `medquad_ans_0004725-1`
**Unsupported citations**: _none_

## Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.0323) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound biotinidase deficiency,...
2. `medquad_ans_0000125-1` (score=0.0246) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few months of life, alth...
3. `medquad_ans_0000125-2` (score=0.0221) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
4. `medquad_ans_0003264-2` (score=0.0194) - What are the signs and symptoms of Isobutyryl-CoA dehydrogenase deficiency? Infants with IBD deficiency usually appear healthy at birth. The signs and symptoms of IBD deficiency may not appear until later in infancy or childhood and can ...
5. `medquad_ans_0000125-3` (score=0.0164) - Mutations in the BTD gene cause biotinidase deficiency. The BTD gene provides instructions for making an enzyme called biotinidase. This enzyme recycles biotin, a B vitamin found in foods such as liver, egg yolks, and milk. Biotinidase r...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Top docs**:

1. `medquad_ans_0000019-1__dup3` (score=0.0283) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air pollution in cities. ...
2. `medquad_ans_0000298-1__dup1` (score=0.0283) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includes rivers, lakes and...
3. `medquad_ans_0000366-7` (score=0.0250) - Prevention No vaccine is available. The best way to prevent schistosomiasis is to take the following steps if you are visiting or live in an area where schistosomiasis is transmitted: - Avoid swimming or wading in freshwater when you are...
4. `medquad_ans_0000368-1__dup1` (score=0.0245) - Floods are common in the United States. Weather such as heavy rain, thunderstorms, hurricanes, or tsunamis can cause flooding. Flooding can also happen when a river or stream overflows its bank, when a levee is breached, or when a dam br...
5. `medquad_ans_0000199-1__dup2` (score=0.0198) - You have two kidneys, each about the size of your fist. Their main job is to filter wastes and excess water out of your blood to make urine. They also keep the body's chemical balance, help control blood pressure, and make hormones. Chro...

**Answer**:

The provided context does not contain specific information about water pollution. While it discusses drinking water sources, treatment, and safety, it does not address water pollution directly [medquad_ans_0000298-1__dup1]. Therefore, based on the given context, there is no information about water pollution.

**Citations detected**: `medquad_ans_0000298-1__dup1`
**Supported citations**: `medquad_ans_0000298-1__dup1`
**Unsupported citations**: _none_

## Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.0328) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue ...
2. `medquad_ans_0000132-1__dup1` (score=0.0259) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend their knees to lift t...
3. `medquad_ans_0000132-2__dup1` (score=0.0239) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the muscles and maintain jo...
4. `medquad_ans_0000222-3__dup1` (score=0.0161) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
5. `medquad_ans_0005812-1` (score=0.0159) - Split hand split foot nystagmus is a rare congenital syndrome characterized by split hand and split foot deformity and eye abnormalities, especially nystagmus. It is thought to have an autosomal dominant mode of inheritance. Currently, t...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Citations detected**: `medquad_ans_0000132-3__dup1`
**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

## Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Top docs**:

1. `medquad_ans_0000915-1__dup1` (score=0.0320) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
2. `medquad_ans_0000916-1__dup1` (score=0.0320) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
3. `medquad_ans_0000916-5__dup1` (score=0.0318) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene known to be associa...
4. `medquad_ans_0000916-3__dup1` (score=0.0313) - What causes Camurati-Engelmann disease? Mutations in the TGFB1 gene cause Camurati-Engelmann disease. The TGFB1 gene provides instructions for producing a protein called transforming growth factor beta-1 (TGF-1). The TGF-1 protein helps ...
5. `medquad_ans_0000145-1` (score=0.0303) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also af...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs, and in some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people develop pain or weakness by adolescence. The disease is caused by a mutation in the TGFB1 gene, which is inherited in an autosomal dominant fashion. In some instances, individuals have the gene mutation but do not develop characteristic features, and in others, features are present but no mutation can be identified—these cases are referred to as Camurati-Engelmann disease type II. Diagnosis is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.0325) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the information is availab...
2. `medquad_ans_0003612-1` (score=0.0295) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2G. If the information is available, the ...
3. `medquad_ans_0004225-2` (score=0.0265) - What are the signs and symptoms of Muscular dystrophy, congenital, merosin-positive? The Human Phenotype Ontology provides the following list of signs and symptoms for Muscular dystrophy, congenital, merosin-positive. If the information ...
4. `medquad_ans_0001935-2` (score=0.0246) - What are the signs and symptoms of Duchenne muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Duchenne muscular dystrophy. If the information is available, the table below includes how...
5. `medquad_ans_0004224-1` (score=0.0230) - What are the signs and symptoms of Muscular dystrophy, congenital, megaconial type? The Human Phenotype Ontology provides the following list of signs and symptoms for Muscular dystrophy, congenital, megaconial type. If the information is...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Citations detected**: `medquad_ans_0002080-2`
**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

## Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Top docs**:

1. `medquad_ans_0000072-5` (score=0.0323) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins McKusick-Nathans Insti...
2. `medquad_ans_0000072-3` (score=0.0164) - Arterial tortuosity syndrome is caused by mutations in the SLC2A10 gene. This gene provides instructions for making a protein called GLUT10. The level of GLUT10 appears to be involved in the regulation of a process called the transformin...
3. `medquad_ans_0000072-1` (score=0.0161) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastrointestinal tract. As i...
4. `medquad_ans_0001030-5` (score=0.0161) - These resources address the diagnosis or management of Weaver syndrome: - Genetic Testing Registry: Weaver syndrome These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Dia...
5. `medquad_ans_0000871-3__dup1` (score=0.0159) - How might Budd-Chiari syndrome be treated? The treatment of Budd-Chiari syndrome varies, depending on the cause of the blockage. Medical treatments may include: Blood-thinning (anticoagulation) medications Clot-busting drugs (thrombolyti...

**Answer**:

The provided context does not specify treatments for arterial tortuosity syndrome. While it mentions resources related to diagnosis and management of the condition, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and information on aneurysm treatment, it does not detail specific treatments for arterial tortuosity syndrome itself [medquad_ans_0000072-5]. Therefore, based on the available information, the exact treatments are not described.

**Citations detected**: `medquad_ans_0000072-5`
**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

## Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.0299) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is available, the table be...
2. `medquad_ans_0001093-1__dup1` (score=0.0297) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is available, the table be...
3. `medquad_ans_0001094-2__dup1` (score=0.0279) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is available, the table be...
4. `medquad_ans_0001106-1` (score=0.0272) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2H? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2H. If the information is available, the table be...
5. `medquad_ans_0000559-1__dup1` (score=0.0272) - What are the signs and symptoms of Autosomal dominant intermediate Charcot-Marie-Tooth disease type C? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant intermediate Charcot-Marie-Tooth...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1].

**Citations detected**: `medquad_ans_0001092-1__dup1`
**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

## Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.0325) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm with supravalvular ...
2. `medquad_ans_0000870-1` (score=0.0320) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS damages the arteries ...
3. `medquad_ans_0002946-1` (score=0.0258) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...
4. `medquad_ans_0005115-2` (score=0.0258) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how often the symptom is ...
5. `medquad_ans_0003723-1` (score=0.0256) - What are the signs and symptoms of Macrocephaly mesodermal hamartoma spectrum? The Human Phenotype Ontology provides the following list of signs and symptoms for Macrocephaly mesodermal hamartoma spectrum. If the information is available...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis. Retinal arterial macroaneurysms can rupture, leading to bleeding within the eye that may cause vision loss. Additionally, supravalvular pulmonic stenosis results in narrowing of the pulmonary artery above the pulmonic valve, impairing blood flow to the lungs and reducing oxygenation of the blood. This leads to symptoms such as shortness of breath. [medquad_ans_0005321-1][medquad_ans_0000870-1]

**Citations detected**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Supported citations**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Unsupported citations**: _none_

## Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.0325) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is caused by mutations...
2. `medquad_ans_0001838-2` (score=0.0325) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies were desc...
3. `medquad_ans_0001819-1` (score=0.0250) - What are the signs and symptoms of D-glycericacidemia? The Human Phenotype Ontology provides the following list of signs and symptoms for D-glycericacidemia. If the information is available, the table below includes how often the symptom...
4. `medquad_ans_0002894-1` (score=0.0240) - What are the signs and symptoms of Histidinuria renal tubular defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Histidinuria renal tubular defect. If the information is available, the table below ...
5. `medquad_ans_0006260-2` (score=0.0240) - What are the signs and symptoms of Tyrosinemia type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tyrosinemia type 1. If the information is available, the table below includes how often the symptom...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and inherited in an autosomal recessive fashion. There are no common signs or symptoms; however, in individual cases, hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies have been reported. Some individuals have been diagnosed with virtually no signs or symptoms [medquad_ans_0001838-1][medquad_ans_0001838-2].

**Citations detected**: `medquad_ans_0001838-1`, `medquad_ans_0001838-2`
**Supported citations**: `medquad_ans_0001838-1`, `medquad_ans_0001838-2`
**Unsupported citations**: _none_

## Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Top docs**:

1. `medquad_ans_0005860-3` (score=0.0164) - Is spondylothoracic dysostosis genetic? Yes. Spondylothoracic dysostosis is caused by mutations in the MESP2 gene. It is inherited in an autosomal recessive fashion.
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0005860-1` (score=0.0161) - Spondylothoracic dysostosis is a rare condition that affects the bones of the spine and the ribs. Signs and symptoms are generally present at birth and may include short-trunk dwarfism (a short body with normal length arms and legs); a s...
4. `medquad_ans_0001082-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000949-1` (score=0.0159) - Spondylothoracic dysostosis is a condition characterized by the malformation of the bones of the spine and ribs. The bones of the spine (vertebrae) do not develop properly, which causes them to be misshapen and abnormally joined together...

**Answer**:

Yes, spondylothoracic dysostosis is inherited. It is caused by mutations in the MESP2 gene and is inherited in an autosomal recessive manner [medquad_ans_0005860-3]. This means both copies of the gene in each cell have mutations, and each parent of an affected individual carries one copy of the mutated gene without showing signs or symptoms [medquad_ans_0000785-4][medquad_ans_0001082-4].

**Citations detected**: `medquad_ans_0005860-3`, `medquad_ans_0000785-4`, `medquad_ans_0001082-4`
**Supported citations**: `medquad_ans_0005860-3`, `medquad_ans_0000785-4`, `medquad_ans_0001082-4`
**Unsupported citations**: _none_

## Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Top docs**:

1. `medquad_ans_0003296-1` (score=0.0320) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. It is inherited in an...
2. `medquad_ans_0000542-5` (score=0.0320) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome - MedlinePlus Encycl...
3. `medquad_ans_0003296-2` (score=0.0313) - What are the signs and symptoms of Jervell Lange-Nielsen syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Jervell Lange-Nielsen syndrome. If the information is available, the table below includ...
4. `medquad_ans_0003295-1` (score=0.0313) - What are the signs and symptoms of Jervell and Lange-Nielsen syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Jervell and Lange-Nielsen syndrome 2. If the information is available, the table ...
5. `medquad_ans_0000542-3` (score=0.0313) - Mutations in the KCNE1 and KCNQ1 genes cause Jervell and Lange-Nielsen syndrome. The KCNE1 and KCNQ1 genes provide instructions for making proteins that work together to form a channel across cell membranes. These channels transport posi...

**Answer**:

Jervell Lange-Nielsen syndrome is a form of long QT syndrome characterized by congenital sensorineural hearing impairment, a prolonged QT interval, syncope, torsade de pointes, and sudden cardiac death. It is inherited in an autosomal recessive fashion and includes two types: type 1 and type 2 [medquad_ans_0003296-1][medquad_ans_0003296-2].

**Citations detected**: `medquad_ans_0003296-1`, `medquad_ans_0003296-2`
**Supported citations**: `medquad_ans_0003296-1`, `medquad_ans_0003296-2`
**Unsupported citations**: _none_

## Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Top docs**:

1. `medquad_ans_0000218-1` (score=0.0260) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and other eye abnormali...
2. `medquad_ans_0000218-5` (score=0.0164) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Med...
3. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0000650-2` (score=0.0161) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
5. `medquad_ans_0003571-1` (score=0.0159) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropathy . If the informat...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy. [medquad_ans_0000218-1] describes the disorder's features and complications but does not include epidemiological data on prevalence or affected population size. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000218-1`
**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

## Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Top docs**:

1. `medquad_ans_0000040-3` (score=0.0325) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although the exact function of ...
2. `medquad_ans_0000040-4` (score=0.0315) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the gene in each cell is ...
3. `medquad_ans_0000259-1__dup1` (score=0.0280) - Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is a genetic condition that causes intellectual disability, muscle weakness (hypotonia), short height, a particular facial appearance, genital abnormalities, and possibly...
4. `medquad_ans_0000040-1` (score=0.0164) - Alpha thalassemia X-linked intellectual disability syndrome is an inherited disorder that affects many parts of the body. This condition occurs almost exclusively in males. Males with alpha thalassemia X-linked intellectual disability sy...
5. `medquad_ans_0000040-5` (score=0.0159) - These resources address the diagnosis or management of alpha thalassemia X-linked intellectual disability syndrome: - Gene Review: Gene Review: Alpha-Thalassemia X-Linked Intellectual Disability Syndrome - Genetic Testing Registry: ATR-X...

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene [medquad_ans_0000040-3]. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression [medquad_ans_0000040-3]. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia [medquad_ans_0000040-3]. Abnormal expression of other genes, whose identities have not been identified, probably leads to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3]. The condition is inherited in an X-linked recessive pattern, with the ATRX gene located on the X chromosome [medquad

**Citations detected**: `medquad_ans_0000040-3`
**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

## Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Top docs**:

1. `medquad_ans_0002615-2` (score=0.0313) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have difficulty exercising ...
2. `medquad_ans_0002622-2` (score=0.0311) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, the table below incl...
3. `medquad_ans_0002623-2` (score=0.0289) - What are the signs and symptoms of Glycogen storage disease type 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 7. If the information is available, the table below incl...
4. `medquad_ans_0002614-1` (score=0.0285) - What are the signs and symptoms of Glycogen storage disease type 12? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 12. If the information is available, the table below in...
5. `medquad_ans_0002617-2` (score=0.0277) - What are the signs and symptoms of Glycogen storage disease type 1B? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 1B. If the information is available, the table below in...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (90%), short stature (90%), autosomal recessive inheritance, hepatomegaly, increased hepatic glycogen content, and postnatal growth retardation [medquad_ans_0002622-2].

**Citations detected**: `medquad_ans_0002622-2`
**Supported citations**: `medquad_ans_0002622-2`
**Unsupported citations**: _none_

## Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Top docs**:

1. `medquad_ans_0000015-2__dup2` (score=0.0296) - Diabetes is a serious, life-long disease. It can lead to problems such as heart disease, stroke, vision loss, kidney disease, and nerve damage. More than 8 million people in the United States have type 2 diabetes and dont know it. Many p...
2. `medquad_ans_0000015-17` (score=0.0278) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone who does not have diab...
3. `medquad_ans_0004434-2` (score=0.0270) - What causes nonalcoholic steatohepatitis? The underlying cause of NASH remains unclear. It most often occurs in persons who are middle-aged and overweight or obese. Many patients with NASH have elevated blood lipids, such as cholesterol ...
4. `medquad_ans_0004342-3` (score=0.0261) - What causes nephrogenic diabetes insipidus? Nephrogenic diabetes insipidus can be either acquired or hereditary. The acquired form can result from chronic kidney disease, certain medications (such as lithium), low levels of potassium in ...
5. `medquad_ans_0000015-16` (score=0.0246) - Type 1 diabetes is an autoimmune disease. In an autoimmune reaction, antibodies, or immune cells, attach to the body's own healthy tissues by mistake, signaling the body to attack them. At present, scientists do not know exactly what cau...

**Answer**:

The causes of diabetes are not fully understood, but for Type 1 diabetes, it is believed that both genetic factors and environmental factors, such as viruses, are involved in the autoimmune reaction where the body's immune system attacks its own healthy tissues [medquad_ans_0000015-2__dup2]. For Type 2 diabetes, while the exact cause is not known, it is closely linked to overweight and obesity, high blood pressure, abnormal cholesterol levels, and genetic factors—having certain genes or combinations of genes may increase or decrease a person's risk for developing the disease [medquad_ans_0000015-2__dup2].

**Citations detected**: `medquad_ans_0000015-2__dup2`
**Supported citations**: `medquad_ans_0000015-2__dup2`
**Unsupported citations**: _none_

## Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-4__dup2` (score=0.0303) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels...
2. `medquad_ans_0000006_3-8` (score=0.0288) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
3. `medquad_ans_0000024_2-8` (score=0.0255) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000031_1-8` (score=0.0240) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
5. `medquad_ans_0000019_3-7` (score=0.0237) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, taking part in a clini...

**Answer**:

The provided context does not mention any ongoing research or clinical trials for diabetic retinopathy. Therefore, based on the given information, no specific research or clinical trials are cited or described. [medquad_ans_0000016-4__dup2]

**Citations detected**: `medquad_ans_0000016-4__dup2`
**Supported citations**: `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

## Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.0328) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart and lung diseases wors...
2. `medquad_ans_0000136-1__dup3` (score=0.0262) - A burn is damage to your body's tissues caused by heat, chemicals, electricity, sunlight or radiation. Scalds from hot liquids and steam, building fires and flammable liquids and gases are the most common causes of burns. Another kind is...
3. `medquad_ans_0000977-1__dup1` (score=0.0261) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from minor to life-threate...
4. `medquad_ans_0000854-3__dup1` (score=0.0259) - What causes bronchiolitis obliterans organizing pneumonia (BOOP)? BOOP may be caused by a variety of factors, including viral infections, inhalation of toxic gases, drugs, connective tissue disorders, radiation therapy, cocaine, inflamma...
5. `medquad_ans_0000774-1__dup1` (score=0.0161) - Respiratory failure happens when not enough oxygen passes from your lungs into your blood. Your body's organs, such as your heart and brain, need oxygen-rich blood to work well. Respiratory failure also can happen if your lungs can't rem...

**Answer**:

An inhalation injury is damage to the body caused by breathing smoke or toxic fumes, which can affect the eyes and respiratory system. These injuries may worsen chronic heart and lung diseases and present with symptoms such as coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. Inhalation injuries are one type of burn, caused by breathing smoke, and can lead to swelling, blistering, scarring, shock, and even death in serious cases [medquad_ans_0000136-1\_\_dup3][medquad_ans_0000516-1__dup1].

**Citations detected**: `medquad_ans_0000136-1__dup3`, `medquad_ans_0000516-1__dup1`
**Supported citations**: `medquad_ans_0000136-1__dup3`, `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

## Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.0320) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has a parent with MEN1 i...
2. `medquad_ans_0000134-10` (score=0.0273) - - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. - Experts do not know the exact cause of Zollinger-Ellison syndrome. - About 25 to 30 percent of gastrinomas are cause...
3. `medquad_ans_0000134-9` (score=0.0247) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
4. `medquad_ans_0000134-8` (score=0.0161) - A health care provider treats Zollinger-Ellison syndrome with medications to reduce gastric acid secretion and with surgery to remove gastrinomas. A health care provider sometimes uses chemotherapymedications to shrink tumorswhen tumors ...
5. `medquad_ans_0000359-2` (score=0.0161) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.

**Answer**:

Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Citations detected**: `medquad_ans_0000134-4__dup2`
**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

## Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.0294) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Genetic Testing Regist...
2. `medquad_ans_0000445-3__dup1` (score=0.0242) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). This may be accomplis...
3. `medquad_ans_0000974-3` (score=0.0164) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiopterin in the body. T...
4. `medquad_ans_0000974-2` (score=0.0161) - This condition is rare, affecting an estimated 1 in 500,000 to 1 in 1 million newborns. In most parts of the world, tetrahydrobiopterin deficiency accounts for 1 to 3 percent of all cases of elevated phenylalanine levels. The remaining c...
5. `medquad_ans_0000124-5` (score=0.0161) - These resources address the diagnosis or management of biotin-thiamine-responsive basal ganglia disease: - Gene Review: Gene Review: Biotin-Thiamine-Responsive Basal Ganglia Disease These resources from MedlinePlus offer information abou...

**Answer**:

The provided context does not specify the treatments for tetrahydrobiopterin deficiency. While it describes the genetic basis and the role of tetrahydrobiopterin in processing phenylalanine and neurotransmitter production, no information is given about specific therapeutic interventions such as medications, dietary changes, or management strategies for this condition [medquad_ans_0000974-5][medquad_ans_0000974-3]. Therefore, based on the available information, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Citations detected**: `medquad_ans_0000974-5`, `medquad_ans_0000974-3`
**Supported citations**: `medquad_ans_0000974-5`, `medquad_ans_0000974-3`
**Unsupported citations**: _none_

## Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.0310) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000736-1` (score=0.0164) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs when the brain fails ...
3. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0000736-3` (score=0.0161) - Mutations in 11 genes have been found to cause nonsyndromic holoprosencephaly. These genes provide instructions for making proteins that are important for normal embryonic development, particularly for determining the shape of the brain ...
5. `medquad_ans_0000650-2` (score=0.0161) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Nonsyndromic holoprosencephaly affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2]. This represents approximately 25 to 50 percent of all cases of holoprosencephaly [medquad_ans_0000736-2]. Therefore, the prevalence of nonsyndromic holoprosencephaly is about 1 in 10,000 newborns.

**Citations detected**: `medquad_ans_0000736-2`
**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

## Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0318) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt export pump (BSEP). Th...
2. `medquad_ans_0000525-4` (score=0.0305) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorder. Some women with a...
3. `medquad_ans_0000118-3` (score=0.0267) - Mutations in the ATP8B1 gene cause benign recurrent intrahepatic cholestasis type 1 (BRIC1), and mutations in the ABCB11 gene cause benign recurrent intrahepatic cholestasis type 2 (BRIC2). These two genes are involved in the release (se...
4. `medquad_ans_0000829-5` (score=0.0164) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholestasis 2 - Genetic Tes...
5. `medquad_ans_0000118-1` (score=0.0161) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called bile. Because the pr...

**Answer**:

The provided context does not mention progressive familial intrahepatic cholestasis or its associated genetic changes. Therefore, based on the given information, it is not possible to answer the question. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.0328) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below includes how often the...
2. `medquad_ans_0004193-1` (score=0.0290) - What are the signs and symptoms of Multiple epiphyseal dysplasia 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple epiphyseal dysplasia 4. If the information is available, the table below incl...
3. `medquad_ans_0003961-1` (score=0.0245) - What are the signs and symptoms of Mesomelic dysplasia Savarirayan type? The Human Phenotype Ontology provides the following list of signs and symptoms for Mesomelic dysplasia Savarirayan type. If the information is available, the table ...
4. `medquad_ans_0005848-1` (score=0.0237) - What are the signs and symptoms of Spondylometaphyseal dysplasia Algerian type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia Algerian type. If the information is availab...
5. `medquad_ans_0005857-1` (score=0.0231) - What are the signs and symptoms of Spondylometaphyseal dysplasia X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia X-linked. If the information is available, the ta...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Citations detected**: `medquad_ans_0005678-1`
**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

## Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.0164) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive disease is a condit...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000453-5` (score=0.0161) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodeficiency These resources...
4. `medquad_ans_0000131-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0001517-2` (score=0.0159) - What causes congenital hepatic fibrosis? Isolated congenital hepatic fibrosis is rare. Congenital hepatic fibrosis is usually associated with conditions known as hepatorenal fibrocystic diseases (FCD) that can also affect the kidneys. Ex...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency (VODI) is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs and symptoms [medquad_ans_0000453-1][medquad_ans_0000785-4][medquad_ans_0000131-4].

**Citations detected**: `medquad_ans_0000453-1`, `medquad_ans_0000785-4`, `medquad_ans_0000131-4`
**Supported citations**: `medquad_ans_0000453-1`, `medquad_ans_0000785-4`, `medquad_ans_0000131-4`
**Unsupported citations**: _none_

## Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.0328) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how often the symptom is ...
2. `medquad_ans_0000057-22` (score=0.0161) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions. - When did I first ...
3. `medquad_ans_0001768-1` (score=0.0161) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is available, the table ...
4. `medquad_ans_0000036-19` (score=0.0159) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood pressure under adequate c...
5. `medquad_ans_0005210-2` (score=0.0159) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how often the symptom is se...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles information from sources such as Orphanet, a European rare disease database [medquad_ans_0005902-1].

**Citations detected**: `medquad_ans_0005902-1`
**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

## Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Top docs**:

1. `medquad_ans_0000159-6` (score=0.0320) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephritis develop kidney f...
2. `medquad_ans_0000159-7` (score=0.0318) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may need to make to their...
3. `medquad_ans_0000159-8` (score=0.0315) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed through urine and b...
4. `medquad_ans_0003688-1` (score=0.0313) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, high blood pressure, ...
5. `medquad_ans_0000159-5__dup1` (score=0.0313) - Lupus nephritis is treated with medications that suppress the immune system, so it stops attacking and damaging the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys. An immu...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. People with lupus nephritis causing high blood pressure may also need blood pressure-lowering medications to slow the progression of kidney disease [medquad_ans_0000159-8]. Early diagnosis and treatment are crucial to prevent chronic kidney damage [medquad_ans_0000159-6]. Additionally, individuals with kidney disease progressing due to lupus nephritis may need to consult a healthcare provider about dietary changes, such as reducing sodium intake if they have high blood pressure [medquad

**Citations detected**: `medquad_ans_0000159-8`, `medquad_ans_0000159-6`
**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-6`
**Unsupported citations**: _none_

## Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Top docs**:

1. `medquad_ans_0000869-5` (score=0.0325) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndrome, susceptibility t...
2. `medquad_ans_0000776-1__dup2` (score=0.0290) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation. Moving makes your ...
3. `medquad_ans_0005317-1` (score=0.0284) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sitting or lying down a...
4. `medquad_ans_0000246-1__dup2` (score=0.0271) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the disorder often desc...
5. `medquad_ans_0000114-2__dup3` (score=0.0249) - Faulty Use of Iron or Lack of Iron Research suggests that the main cause of restless legs syndrome (RLS) is a faulty use of iron or a lack of iron in the brain. The brain uses iron to make the chemical dopamine (DO-pah-meen) and to contr...

**Answer**:

Treatments for Restless Legs Syndrome include lifestyle changes such as regular sleep habits, relaxation techniques, and moderate exercise during the day. If these do not work, medicines may be used to reduce symptoms [medquad_ans_0000776-1__dup2]. The National Heart, Lung, and Blood Institute provides information on how RLS is treated, and the Restless Leg Syndrome Foundation offers treatment options [medquad_ans_0000869-5]. Additionally, treatment may be based on the individual's specific condition, with some people also experiencing periodic limb movement disorder (PLMD), which may influence management strategies [medquad_ans_0005317-1].

**Citations detected**: `medquad_ans_0000776-1__dup2`, `medquad_ans_0000869-5`, `medquad_ans_0005317-1`
**Supported citations**: `medquad_ans_0000776-1__dup2`, `medquad_ans_0000869-5`, `medquad_ans_0005317-1`
**Unsupported citations**: _none_

## Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Top docs**:

1. `medquad_ans_0002573-6` (score=0.0325) - How might glioblastoma be treated? Unfortunately, there is no cure for glioblastoma. Treatment is palliative and may include surgery, radiation therapy and/or chemotherapy. The best treatment options for each person depend on many factor...
2. `medquad_ans_0002573-2` (score=0.0308) - What are the signs and symptoms of glioblastoma? Signs and symptoms of glioblastoma vary depending on the size and location of the tumor but may include: Headache Nausea and vomiting Drowsiness Changes in personality Weakness on one side...
3. `medquad_ans_0002575-1` (score=0.0301) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enlarge. Signs and symp...
4. `medquad_ans_0002573-1` (score=0.0290) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brai...
5. `medquad_ans_0002573-3` (score=0.0262) - What causes glioblastoma? In most cases, the exact underlying cause of glioblastoma is unknown. In rare cases, they can occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome and Li Fraumeni synd...

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte [medquad_ans_0002573-1]. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue [medquad_ans_0002573-1]. There is currently no cure for glioblastoma, and treatment is palliative and may include surgery, radiation therapy, and/or chemotherapy [medquad_ans_0002573-1][medquad_ans_0002573-6]. In most cases, the exact underlying cause is unknown; however, they can rarely occur in people with certain genetic syndromes such as neurofibromatosis

**Citations detected**: `medquad_ans_0002573-1`, `medquad_ans_0002573-6`
**Supported citations**: `medquad_ans_0002573-1`, `medquad_ans_0002573-6`
**Unsupported citations**: _none_

## Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.0328) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can af...
2. `medquad_ans_0005496-4` (score=0.0311) - Is schizencephaly inherited? Schizencephaly is not thought to be inherited in most cases and it rarely affects more than one person in a family. A few cases of familial schizencephaly have been linked to changes (mutations) in the EMX2 g...
3. `medquad_ans_0000251-3__dup1` (score=0.0308) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
4. `medquad_ans_0005496-1` (score=0.0306) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include developmental delay, seizur...
5. `medquad_ans_0000251-1__dup2` (score=0.0303) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) commonly have developme...

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, supporting a genetic cause in some cases. It has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Citations detected**: `medquad_ans_0005496-3`
**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

## Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0000646-1__dup2` (score=0.0288) - Neuromuscular disorders affect the nerves that control your voluntary muscles. Voluntary muscles are the ones you can control, like in your arms and legs. Your nerve cells, also called neurons, send the messages that control these muscle...
2. `medquad_ans_0000309-3__dup1` (score=0.0284) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the family). The remain...
3. `medquad_ans_0003331-1` (score=0.0276) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
4. `medquad_ans_0000052-1` (score=0.0270) - Amyotrophic lateral sclerosis (ALS) is a progressive disease that affects motor neurons, which are specialized nerve cells that control muscle movement. These nerve cells are found in the spinal cord and the brain. In ALS, motor neurons ...
5. `medquad_ans_0000309-1__dup1` (score=0.0269) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which are distinguished by t...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. These mutations can cause familial ALS and may also contribute to the development of sporadic ALS. Approximately 60% of individuals with familial ALS have an identifiable genetic mutation, while the genetic cause in the remaining cases is unknown. The genes associated with ALS appear to play a role in how neurons function or in regulating the production of various proteins [medquad_ans_0000309-3__dup1].

**Citations detected**: `medquad_ans_0000309-3__dup1`
**Supported citations**: `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

## Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Top docs**:

1. `medquad_ans_0003473-6` (score=0.0308) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injections of insulin-like g...
2. `medquad_ans_0003473-5` (score=0.0164) - How is Laron syndrome diagnosed? A diagnosis of Laron syndrome is often suspected based on the presence of characteristic signs and symptoms. Additional testing can then be ordered to confirm the diagnosis and rule out other conditions t...
3. `medquad_ans_0000581-5` (score=0.0164) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Testing Registry: Laro...
4. `medquad_ans_0003473-2` (score=0.0161) - What are the signs and symptoms of Laron syndrome? Laron syndrome is a rare condition in which the body is unable to use growth hormone. The primary symptom is short stature. Although affected people are generally close to average size a...
5. `medquad_ans_0000581-1` (score=0.0159) - Laron syndrome is a rare form of short stature that results from the body's inability to use growth hormone, a substance produced by the brain's pituitary gland that helps promote growth. Affected individuals are close to normal size at ...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and growth hormone (GH) levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6]. There is currently no cure for Laron syndrome, and treatment is primarily focused on improving growth [medquad_ans_0003473-6].

**Citations detected**: `medquad_ans_0003473-6`
**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

## Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-19` (score=0.0311) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although both treatments have h...
2. `medquad_ans_0000016-4__dup2` (score=0.0310) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels...
3. `medquad_ans_0000016-15` (score=0.0295) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood sugar, blood pressure...
4. `medquad_ans_0000016-21` (score=0.0289) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the laboratory and with pat...
5. `medquad_ans_0000016-24` (score=0.0273) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even blindness. Macular e...

**Answer**:

The treatments for diabetic retinopathy include:

- For macular edema: Prompt treatment with anti-VEGF drugs, with or without laser treatment, results in better vision than laser treatment alone or steroid injections. These drugs reduce fluid leakage and interfere with the growth of new blood vessels in the retina. In some cases, focal laser treatment is used along with eye injections. Focal laser treatment involves placing up to several hundred small laser burns around the macula to slow fluid leakage and reduce fluid in the retina. The procedure is usually completed in one session, and further treatment may be needed [medquad_ans_0000016-4__dup2].

- For proliferative retinopathy: Treatment involves scatter laser treatment, where 1,000 to 2,000 laser burns are placed in areas of the retina away from the macula to shrink abnormal blood vessels. This treatment usually requires two or more sessions

**Citations detected**: `medquad_ans_0000016-4__dup2`
**Supported citations**: `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

## Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Top docs**:

1. `medquad_ans_0001421-3` (score=0.0313) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous remission), although m...
2. `medquad_ans_0001421-1` (score=0.0305) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; the inflammation can...
3. `medquad_ans_0000116-11` (score=0.0297) - - Microscopic colitis is an inflammation of the colon that a health care provider can see only with a microscope. - The two types of microscopic colitis are collagenous colitis and lymphocytic colitis. - The exact cause of microscopic co...
4. `medquad_ans_0001421-2` (score=0.0291) - What are the signs and symptoms of collagenous colitis? All individuals with collagenous colitis experience chronic, watery, non-bloody diarrhea which is what typically prompts individuals to seek medical attention. Onset of diarrhea may...
5. `medquad_ans_0000122-9` (score=0.0284) - Complications of ulcerative colitis can include - rectal bleedingwhen ulcers in the intestinal lining open and bleed. Rectal bleeding can cause anemia, which health care providers can treat with diet changes and iron supplements. People ...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes are not sufficient, medications can be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but are usually only used for sudden attacks of diarrhea; long-term use is typically avoided due to unwanted side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine, and l

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Top docs**:

1. `medquad_ans_0000418-7` (score=0.0306) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause of many cases of tri...
2. `medquad_ans_0000418-2__dup1` (score=0.0305) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
3. `medquad_ans_0000689-1__dup2` (score=0.0240) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are easily treated and som...
4. `medquad_ans_0000241-7` (score=0.0164) - The best way to avoid hookworm infection is not to walk barefoot in areas where hookworm is common and where there may be human fecal contamination of the soil. Also, avoid other skin contact with such soil and avoid ingesting it. Infect...
5. `medquad_ans_0000808-3` (score=0.0161) - Mutations in the GAA gene cause Pompe disease. The GAA gene provides instructions for producing an enzyme called acid alpha-glucosidase (also known as acid maltase). This enzyme is active in lysosomes, which are structures that serve as ...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), follow these measures:

- Wash your hands with warm water and soap after handling raw meat [medquad_ans_0000418-7].
- Ensure pork is frozen for at least 20 days at 5°F (-15°C) if it is less than 6 inches thick to kill any Trichinella worms [medquad_ans_0000418-7].
- Freezing wild game meats may not effectively kill all worms, as some worm species in wild animals are freeze-resistant; therefore, avoid consuming undercooked wild game meat, especially bear meat [medquad_ans_0000418-7][medquad_ans_0000418-2__dup1].
- Do not allow pigs or wild animals to eat uncooked meat, scraps, or carcasses

**Citations detected**: `medquad_ans_0000418-7`, `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-7`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Top docs**:

1. `medquad_ans_0000536-3` (score=0.0325) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic attack. This autoimmun...
2. `medquad_ans_0000182-5__dup1` (score=0.0310) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood. About 70 percent o...
3. `medquad_ans_0000536-1` (score=0.0310) - Autoimmune hepatitis is a disease in which the bodys immune system attacks liver cells. This immune response causes inflammation of the liver, also called hepatitis. The disease can be quite serious and, if not treated, gets worse over t...
4. `medquad_ans_0000182-1__dup3` (score=0.0308) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally protects people from inf...
5. `medquad_ans_0000182-11` (score=0.0302) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time if not treated. Aut...

**Answer**:

Autoimmune hepatitis is caused by a cell-mediated immunologic attack by the body's immune system on liver cells, which leads to liver inflammation (hepatitis) [medquad_ans_0000536-3]. This autoimmune attack may be triggered by genetic factors, viral infections, or chemical agents [medquad_ans_0000536-3]. The occurrence of autoimmune hepatitis in relatives of individuals with autoimmune diseases further suggests a genetic component [medquad_ans_0000536-3].

**Citations detected**: `medquad_ans_0000536-3`
**Supported citations**: `medquad_ans_0000536-3`
**Unsupported citations**: _none_

## Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.0325) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central nervous system contr...
2. `medquad_ans_0000006_3-4` (score=0.0313) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor grows. - The child's a...
3. `medquad_ans_0000006_3-9` (score=0.0313) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treating childhood brain ...
4. `medquad_ans_0000006_3-7` (score=0.0311) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging is the process used...
5. `medquad_ans_0000006_3-5` (score=0.0263) - Tests that examine the brain and spinal cord are used to detect (find) childhood astrocytomas. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of health. This inclu...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Citations detected**: `medquad_ans_0000006_3-1`
**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

## Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Top docs**:

1. `medquad_ans_0000013_3-4__dup2` (score=0.0325) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL1 negative. For more...
2. `medquad_ans_0000013_3-1__dup3` (score=0.0323) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising ...
3. `medquad_ans_0000013_3-2__dup3` (score=0.0308) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check with your doctor if ...
4. `medquad_ans_0000001_4-6` (score=0.0306) - Key Points - After chronic myelogenous leukemia has been diagnosed, tests are done to find out if the cancer has spread. - Chronic myelogenous leukemia has 3 phases. - Chronic phase - Accelerated phase - Blastic phase After chronic myelo...
5. `medquad_ans_0000013_3-1__dup1` (score=0.0303) - Key Points - Myelodysplastic/myeloproliferative neoplasm, unclassifiable, is a disease that has features of both myelodysplastic and myeloproliferative diseases but is not chronic myelomonocytic leukemia, juvenile myelomonocytic leukemia...

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become a type of white blood cell called granulocytes, and some of these stem cells never become mature white blood cells. These immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. The leukemia cells in atypical CML and CML look alike under a microscope, but in atypical CML, a certain chromosome change called the "Philadelphia chromosome" is not present [medquad_ans_0000013_3-1__dup3].

**Citations detected**: `medquad_ans_0000013_3-1__dup3`
**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

## Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.0328) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the characteristic features...
2. `medquad_ans_0000996-4` (score=0.0323) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with an abnormal number of...
3. `medquad_ans_0000997-3` (score=0.0303) - Most cases of trisomy 18 result from having three copies of chromosome 18 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the characteristic features...
4. `medquad_ans_0006215-4` (score=0.0284) - Is trisomy 18 inherited? Most cases of trisomy 18 are not inherited and occur randomly due to errors in the formation of eggs or sperm. If an egg or sperm gains an extra copy of chromosome 18 during cell division and contributes to a pre...
5. `medquad_ans_0000872-3` (score=0.0282) - Mutations in the RB1 gene are responsible for most cases of retinoblastoma. RB1 is a tumor suppressor gene, which means that it normally regulates cell growth and keeps cells from dividing too rapidly or in an uncontrolled way. Most muta...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In rare cases, trisomy 13 can result from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals have two normal copies of chromosome 13 plus an extra copy attached to another chromosome. Additionally, in rare instances, only part of chromosome 13 is

**Citations detected**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

## Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.0313) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - Genetic Testing Regis...
2. `medquad_ans_0004415-3` (score=0.0308) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respiratory problems A cardio...
3. `medquad_ans_0000218-2__dup1` (score=0.0276) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bone marrow transplant...
4. `medquad_ans_0000731-2` (score=0.0164) - Niemann-Pick disease types A and B is estimated to affect 1 in 250,000 individuals. Niemann-Pick disease type A occurs more frequently among individuals of Ashkenazi (eastern and central European) Jewish descent than in the general popul...
5. `medquad_ans_0000731-3` (score=0.0161) - Niemann-Pick disease types A and B is caused by mutations in the SMPD1 gene. This gene provides instructions for producing an enzyme called acid sphingomyelinase. This enzyme is found in lysosomes, which are compartments within cells tha...

**Answer**:

There is no specific treatment for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, gastroenterologists, and learning specialists [medquad_ans_0004415-3]. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies might be helpful for those with type B [medquad_ans_0000218-2__dup1]. However, restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0000218-2__dup1]. There is currently no effective treatment for persons with type A [medquad_ans_0000218-2__dup1].

**Citations detected**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

## Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.0328) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial polydactyly. If the info...
2. `medquad_ans_0003858-1` (score=0.0317) - What are the signs and symptoms of Meckel syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 3. If the information is available, the table below includes how often the...
3. `medquad_ans_0003857-1` (score=0.0315) - What are the signs and symptoms of Meckel syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 2. If the information is available, the table below includes how often the...
4. `medquad_ans_0005020-1` (score=0.0291) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the table below includes how...
5. `medquad_ans_0004592-1` (score=0.0272) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table below includes how o...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.0323) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Many mantle cell lympho...
2. `medquad_ans_0003789-1` (score=0.0323) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgkin lymphomas and is ...
3. `medquad_ans_0003789-4` (score=0.0320) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be performed, but the diagn...
4. `medquad_ans_0003789-2` (score=0.0315) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night sweats, unexplained high...
5. `medquad_ans_0003789-5` (score=0.0308) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance stagestreatment includ...

**Answer**:

Mantle cell lymphoma is caused by DNA damage within lymphocytes, often due to a chromosomal translocation. Most lymphomas, including mantle cell lymphoma, are acquired diseases rather than inherited, and the DNA damage results from a combination of factors. Specifically, many mantle cell lymphomas are associated with a chromosomal translocation [medquad_ans_0003789-3]. Additionally, environmental factors such as hair dyes and pesticides, as well as immunodeficiencies, have been linked to non-Hodgkin lymphomas, which include mantle cell lymphoma [medquad_ans_0003789-3]. Viral infections like Epstein-Barr virus, HIV, and human herpesvirus 6 have also been associated with non-Hodgkin lymphomas, though specific viral links to mantle cell lymphoma are not detailed in the provided context [medquad_ans_000378

**Citations detected**: `medquad_ans_0003789-3`
**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

## Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Top docs**:

1. `medquad_ans_0000589-3` (score=0.0164) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg-Calv-Perthes diseas...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000589-1` (score=0.0161) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically between ages 4 and ...
4. `medquad_ans_0000487-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000589-2` (score=0.0159) - The incidence of Legg-Calv-Perthes disease varies by population. The condition is most common in white populations, in which it affects an estimated 1 to 3 in 20,000 children under age 15.

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene are involved, and this condition is inherited in an autosomal recessive pattern [medquad_ans_0000589-3][medquad_ans_0000785-4]. However, the majority of cases are not inherited. Therefore, Legg-Calv-Perthes disease is not typically inherited.

**Citations detected**: `medquad_ans_0000589-3`, `medquad_ans_0000785-4`
**Supported citations**: `medquad_ans_0000589-3`, `medquad_ans_0000785-4`
**Unsupported citations**: _none_

## Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Top docs**:

1. `medquad_ans_0000647-1` (score=0.0315) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the substance that insulate...
2. `medquad_ans_0003965-3` (score=0.0315) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for an individual to be ...
3. `medquad_ans_0003965-1` (score=0.0311) - Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially cells of the nervous system. This accumulation results in progressive destruction of white matter of ...
4. `medquad_ans_0000647-2` (score=0.0306) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigrated to Israel from s...
5. `medquad_ans_0003966-1` (score=0.0303) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to saposin B deficiency. If ...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly in the nervous system cells that produce myelin. This accumulation leads to progressive destruction of white matter throughout the nervous system, including the central nervous system (brain and spinal cord) and the peripheral nervous system (nerves connecting the brain and spinal cord to muscles and sensory cells). The disorder results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals lose awareness of their surroundings and become unresponsive. Effects of sulfatide accumulation have also been reported on other organs, most commonly the gallbladder. There are several forms: the late infantile form (affecting 50–60% of cases), which typically appears

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.0328) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the table below includes how...
2. `medquad_ans_0005848-1` (score=0.0268) - What are the signs and symptoms of Spondylometaphyseal dysplasia Algerian type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia Algerian type. If the information is availab...
3. `medquad_ans_0006042-1` (score=0.0193) - What are the signs and symptoms of Tetraamelia with ectodermal dysplasia and lacrimal duct abnormalities? The Human Phenotype Ontology provides the following list of signs and symptoms for Tetraamelia with ectodermal dysplasia and lacrim...
4. `medquad_ans_0002946-1` (score=0.0161) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...
5. `medquad_ans_0005127-2` (score=0.0161) - What are the signs and symptoms of Pseudoachondroplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoachondroplasia. If the information is available, the table below includes how often the sym...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Citations detected**: `medquad_ans_0005135-1`
**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

## Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Top docs**:

1. `medquad_ans_0000251-3` (score=0.0325) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that larger deletions te...
2. `medquad_ans_0000251-1` (score=0.0318) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that sounds like that of ...
3. `medquad_ans_0000251-2` (score=0.0310) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
4. `medquad_ans_0000251-4` (score=0.0310) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typically have no history of...
5. `medquad_ans_0000251-5` (score=0.0295) - These resources address the diagnosis or management of cri-du-chat syndrome: - Cri du Chat Syndrome Support Group (UK): Diagnosis - Cri du Chat Syndrome Support Group (UK): Therapies - Genetic Testing Registry: 5p partial monosomy syndro...

**Answer**:

Cri-du-chat syndrome is a chromosomal condition that results when a piece of chromosome 5 is missing, specifically a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. It is characterized by a high-pitched cry that sounds like a cat's cry, intellectual disability, delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children are born with a heart defect. The size of the deletion varies among affected individuals, with larger deletions generally leading to more severe intellectual disability and developmental delay than smaller deletions. Researchers believe that the loss of specific genes, such as CTNND2, is associated with severe intellectual disability, and the loss of other genes in this region contributes to the

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.0328) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmental delay Seizures D...
2. `medquad_ans_0000718-1__dup1` (score=0.0317) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius...
3. `medquad_ans_0000722-2__dup1` (score=0.0298) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the face, tongue, jaws, and...
4. `medquad_ans_0000719-1__dup1` (score=0.0291) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides of the brain (bilate...
5. `medquad_ans_0000722-1__dup1` (score=0.0283) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of the face, tongue, jaws...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Citations detected**: `medquad_ans_0000719-2__dup1`
**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

## Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.0323) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amin...
2. `medquad_ans_0003414-1` (score=0.0323) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, which results in mildly ...
3. `medquad_ans_0003414-2` (score=0.0318) - What are the signs and symptoms of Konigsmark Knox Hussels syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Konigsmark Knox Hussels syndrome. If the information is available, the table below in...
4. `medquad_ans_0003414-4` (score=0.0315) - Is genetic testing available for Konigsmark Knox Hussels syndrome? GeneTests lists the names of laboratories that are performing genetic testing for Konigsmark Knox Hussels syndrome. To view the contact information for the clinical labor...
5. `medquad_ans_0003384-2` (score=0.0156) - What are the signs and symptoms of Keutel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Keutel syndrome. If the information is available, the table below includes how often the symptom is se...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3]. This mutation is inherited in an autosomal dominant fashion [medquad_ans_0003414-1].

**Citations detected**: `medquad_ans_0003414-3`, `medquad_ans_0003414-1`
**Supported citations**: `medquad_ans_0003414-3`, `medquad_ans_0003414-1`
**Unsupported citations**: _none_

## Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Top docs**:

1. `medquad_ans_0004586-1` (score=0.0317) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the table below includes how...
2. `medquad_ans_0004592-1` (score=0.0315) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table below includes how o...
3. `medquad_ans_0004593-1` (score=0.0306) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table below includes how o...
4. `medquad_ans_0004590-1` (score=0.0306) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table below includes how o...
5. `medquad_ans_0004589-2` (score=0.0292) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings Nodules (bumps) of t...

**Answer**:

The provided context does not contain information about the symptoms of Orofaciodigital syndrome 1. [medquad_ans_0004586-1] and [medquad_ans_0004592-1] discuss Orofaciodigital syndrome 11 and Orofaciodigital syndrome 5, respectively, but there is no mention of Orofaciodigital syndrome 1. Therefore, the requested information is not available in the context.

**Citations detected**: `medquad_ans_0004586-1`, `medquad_ans_0004592-1`
**Supported citations**: `medquad_ans_0004586-1`, `medquad_ans_0004592-1`
**Unsupported citations**: _none_

## Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Top docs**:

1. `medquad_ans_0000286-1__dup2` (score=0.0292) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do. Some supplements c...
2. `medquad_ans_0000694-1__dup1` (score=0.0278) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of medical errors, you ca...
3. `medquad_ans_0000925-1__dup1` (score=0.0258) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about your baby's health, ca...
4. `medquad_ans_0000130-1__dup3` (score=0.0252) - Summary : If you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Instead of breast reconstruction, you could choose to wear a breast form that replaces the breast, wear paddin...
5. `medquad_ans_0000499-1__dup1` (score=0.0234) - Summary : A hysterectomy is surgery to remove a woman's uterus or womb. The uterus is the place where a baby grows when a woman is pregnant. After a hysterectomy, you no longer have menstrual periods and can't become pregnant. Sometimes ...

**Answer**:

Yes, there is information about surgery. Patients should make sure they understand what will happen if they need surgery and should ask questions if they have doubts or concerns. It is also important to tell health care providers about all the medicines and dietary supplements being taken, including over-the-counter drugs, and to inform them of any allergies or bad reactions to anesthesia. Additionally, getting a second opinion about treatment options can help in making informed decisions regarding surgery [medquad_ans_0000694-1__dup1].

**Citations detected**: `medquad_ans_0000694-1__dup1`
**Supported citations**: `medquad_ans_0000694-1__dup1`
**Unsupported citations**: _none_

## Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Top docs**:

1. `medquad_ans_0002935-3` (score=0.0328) - What causes human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-2 is spread by blood transfusions, sexual contact and sharing ne...
2. `medquad_ans_0002935-4` (score=0.0318) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspected or diagnosed since...
3. `medquad_ans_0002935-5` (score=0.0315) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to others. Screening blo...
4. `medquad_ans_0002935-1` (score=0.0313) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may late...
5. `medquad_ans_0002934-3` (score=0.0310) - What causes human T-cell leukemia virus, type 1? Human T-cell leukemia virus, type 1 (HTLV-1) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-1 is spread by blood transfusions, sexual contact and sharing ne...

**Answer**:

Human T-cell leukemia virus type 2 (HTLV-2) is a retroviral infection that affects T cells, a type of white blood cell. It is spread by blood transfusions, sexual contact, and sharing needles, and can also be transmitted from mother to child during birth or breastfeeding. Most people with HTLV-2 remain asymptomatic throughout their lives, although some may later develop neurological problems or chronic lung infections. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition [medquad_ans_0002935-1]. Diagnosis is typically made through blood tests that detect antibodies to the virus, though it is often not suspected or diagnosed due to the lack of symptoms in most cases [medquad_ans_0002935-4].

**Citations detected**: `medquad_ans_0002935-1`, `medquad_ans_0002935-4`
**Supported citations**: `medquad_ans_0002935-1`, `medquad_ans_0002935-4`
**Unsupported citations**: _none_

## Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.0323) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other pa...
2. `medquad_ans_0001152-1` (score=0.0318) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD decrease a child's ab...
3. `medquad_ans_0000758-1__dup2` (score=0.0315) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...
4. `medquad_ans_0001152-2` (score=0.0313) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment for chILD is to rel...
5. `medquad_ans_0000038-1__dup2` (score=0.0308) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chronic cough, rapid breat...

**Answer**:

Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other particles in the air is responsible for some types of interstitial lung diseases. Specific types include black lung disease among coal miners from inhaling coal dust, farmer's lung from inhaling farm dust, asbestosis from inhaling asbestos fibers, siderosis from inhaling iron from mines or welding fumes, and silicosis from inhaling silica dust. Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types of interstitial lung disease have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1\_\_dup

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000645-1` (score=0.0164) - Melnick-Needles syndrome is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also includes otopalatodi...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000645-2` (score=0.0161) - Melnick-Needles syndrome is a rare disorder; fewer than 100 cases have been reported worldwide.
4. `medquad_ans_0000960-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000645-5` (score=0.0159) - These resources address the diagnosis or management of Melnick-Needles syndrome: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Melnick-Needles syndrome These resources from MedlinePlus offer ...

**Answer**:

Yes, Melnick-Needles syndrome is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs and symptoms of the condition [medquad_ans_0000785-4][medquad_ans_0000960-4].

**Citations detected**: `medquad_ans_0000785-4`, `medquad_ans_0000960-4`
**Supported citations**: `medquad_ans_0000785-4`, `medquad_ans_0000960-4`
**Unsupported citations**: _none_

## Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Top docs**:

1. `medquad_ans_0006138-6` (score=0.0323) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlying conditions contri...
2. `medquad_ans_0006138-5` (score=0.0315) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing such as CT scan and b...
3. `medquad_ans_0006138-2` (score=0.0312) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or ...
4. `medquad_ans_0006138-1` (score=0.0311) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of TBM: a congenital fo...
5. `medquad_ans_0006138-3` (score=0.0310) - What causes tracheobronchomalacia? The underlying cause of tracheobronchomalacia (TBM) varies by subtype. Most cases of primary TBM (also called congenital TBM) are caused by genetic conditions that weaken the walls of the airway (specif...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and a secondary form (called acquired TBM) that is usually seen in adults [medquad_ans_0006138-2][medquad_ans_0006138-1]. Some affected people may initially have no signs or symptoms, but the condition is typically progressive and many people will eventually develop characteristic features such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing [medquad_ans_0006138-2].

**Citations detected**: `medquad_ans_0006138-2`, `medquad_ans_0006138-1`
**Supported citations**: `medquad_ans_0006138-2`, `medquad_ans_0006138-1`
**Unsupported citations**: _none_

## Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.0328) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, ...
2. `medquad_ans_0000531-1__dup2` (score=0.0318) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The two halves of the man...
3. `medquad_ans_0000371-1__dup1` (score=0.0281) - What are the signs and symptoms of Ankylosis of teeth? The Human Phenotype Ontology provides the following list of signs and symptoms for Ankylosis of teeth. If the information is available, the table below includes how often the symptom...
4. `medquad_ans_0000880-1__dup2` (score=0.0270) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it may cause - Pain tha...
5. `medquad_ans_0000874-2` (score=0.0161) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Citations detected**: `medquad_ans_0006030-1`
**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

## Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000554-2` (score=0.0164) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0003339-3` (score=0.0161) - How is Kallmann syndrome inherited? Kallmann syndrome (KS) may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example: KS due to mutations in the KAL1 ge...
4. `medquad_ans_0001090-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000554-4` (score=0.0159) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one...

**Answer**:

Yes, Kallmann syndrome can be inherited, and the pattern of inheritance depends on the specific gene involved. It may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner. For example, Kallmann syndrome 1, caused by mutations in the KAL1 (ANOS1) gene, is inherited in an X-linked recessive pattern. Syndromes due to mutations in genes such as FGFR1, PROKR2, PROK2, CHD7, or FGF8 are typically inherited in an autosomal dominant manner, although mutations in PROKR2 and PROK2 can also follow an autosomal recessive pattern. In the majority of cases, however, the family history is negative, indicating sporadic occurrence, though affected individuals may still pass the mutation to their children. The risk for each child to be affected depends on the genetic cause and may be up to 50%

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.0325) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0005020-1` (score=0.0310) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the table below includes how...
3. `medquad_ans_0005022-1` (score=0.0304) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the table below includes how...
4. `medquad_ans_0005970-1` (score=0.0284) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0002199-1` (score=0.0283) - What are the signs and symptoms of Eyebrows duplication of, with stretchable skin and syndactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Eyebrows duplication of, with stretchable skin and synda...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Citations detected**: `medquad_ans_0005967-2`
**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

## Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.0323) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tightness. There are tw...
2. `medquad_ans_0000021-4__dup2` (score=0.0318) - Bronchitis is a very common condition. Millions of cases occur every year. Elderly people, infants, and young children are at higher risk for acute bronchitis than people in other age groups. People of all ages can develop chronic bronch...
3. `medquad_ans_0000021-1__dup3` (score=0.0311) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Conditions Index How th...
4. `medquad_ans_0000021-7` (score=0.0310) - The main goals of treating acute and chronic bronchitis are to relieve symptoms and make breathing easier. If you have acute bronchitis, your doctor may recommend rest, plenty of fluids, and aspirin (for adults) or acetaminophen to treat...
5. `medquad_ans_0000021-5__dup1` (score=0.0306) - Acute Bronchitis Acute bronchitis caused by an infection usually develops after you already have a cold or the flu. Symptoms of a cold or the flu include sore throat, fatigue (tiredness), fever, body aches, stuffy or runny nose, vomiting...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs, characterized by a persistent cough that produces mucus. It is one type of chronic obstructive pulmonary disease (COPD) and is often caused by cigarette smoking, as well as long-term exposure to air pollution, fumes, or dust. Symptoms include a chronic cough with mucus, shortness of breath, wheezing, and chest tightness. It is a long-term condition that tends to recur or never fully resolves. Diagnosis involves evaluating symptoms, listening to breathing, and possibly additional tests. Treatment includes medications to open airways and clear mucus, oxygen therapy if needed, and pulmonary rehabilitation to improve daily management. People over 45, especially smokers, are at higher risk, and women are more than twice as likely as men to be diagnosed with chronic bronchitis [medquad_ans_0000

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Top docs**:

1. `medquad_ans_0000934-5` (score=0.0286) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - MedlinePlus Encyclopedi...
2. `medquad_ans_0000936-5` (score=0.0285) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscular Atrophy Types Thes...
3. `medquad_ans_0000935-5` (score=0.0276) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Testing Registry: Adult...
4. `medquad_ans_0000173-4__dup1` (score=0.0272) - Between 2003 and 2012, the NINDS piloted the Spinal Muscular Atrophy Project to expedite therapeutics development for this hereditary neurodegenerative disease. The Project was designed to accelerate the research process by identifying d...
5. `medquad_ans_0000840-1__dup2` (score=0.0262) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms and legs. As the neu...

**Answer**:

The provided context does not specify the exact treatments for Spinal Muscular Atrophy (SMA). However, it mentions that resources from MedlinePlus offer information on diagnosis and management of various health conditions, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000934-5, medquad_ans_0000935-5, medquad_ans_0000936-5]. These categories may include treatment options, but specific therapies for SMA are not detailed in the context. Therefore, based on the given information, the treatments for SMA cannot be explicitly identified.

**Citations detected**: `medquad_ans_0000934-5, medquad_ans_0000935-5, medquad_ans_0000936-5`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000934-5, medquad_ans_0000935-5, medquad_ans_0000936-5`

## Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Top docs**:

1. `medquad_ans_0000001_1-7` (score=0.0325) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - Chemotherapy with ste...
2. `medquad_ans_0000001_6-7` (score=0.0315) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood leukemia. - Children...
3. `medquad_ans_0000040-5__dup1` (score=0.0277) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment for leukemia is to de...
4. `medquad_ans_0000040-19` (score=0.0258) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemia. Learn more about ...
5. `medquad_ans_0000001_2-7` (score=0.0256) - Key Points - There are different types of treatment for patients with adult acute myeloid leukemia. - The treatment of adult AML usually has 2 phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - Stem...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. Additionally, patients may consider participating in clinical trials, which are research studies testing new or improved treatments. [medquad_ans_0000001_1-7]

**Citations detected**: `medquad_ans_0000001_1-7`
**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

## Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.0318) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleot...
2. `medquad_ans_0002439-2` (score=0.0311) - What are the signs and symptoms of Friedreich ataxia? Symptoms usually begin between the ages of 5 and 15 but can, on occasion, appear in adulthood or even as late as age 75. The first symptom to appear is usually difficulty in walking, ...
3. `medquad_ans_0000388-1` (score=0.0305) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of this condition incl...
4. `medquad_ans_0002439-1` (score=0.0301) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features include the gradual...
5. `medquad_ans_0002439-4` (score=0.0301) - How is Friedreich ataxia inherited? Friedreich ataxia is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. The parents of an aff...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Citations detected**: `medquad_ans_0002439-3`
**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

## Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Top docs**:

1. `medquad_ans_0000341-1` (score=0.0323) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which causes the heartbeat to...
2. `medquad_ans_0000341-2` (score=0.0318) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with age. The incidence ...
3. `medquad_ans_0000341-4` (score=0.0318) - Familial atrial fibrillation appears to be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0002230-1` (score=0.0315) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular response in the heart's low...
5. `medquad_ans_0002230-3` (score=0.0310) - How might familial atrial fibrillation be treated? We are unaware of treatment recommendations specific to familial atrial fibrillation, but there is information available about treatment for atrial fibrillation in general. Treatment for...

**Answer**:

Familial atrial fibrillation is an inherited heart condition that disrupts the heart's normal rhythm by causing uncoordinated or erratic electrical activity in the heart's upper chambers (the atria), leading to a fast and irregular heartbeat (arrhythmia). This condition may be caused by changes (mutations) in various genes, some of which have not been identified. It appears to be inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder, although autosomal recessive inheritance has also been reported. The condition increases the risk of stroke and sudden death and may present with symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting. Complications can occur at any age, though some individuals may never experience associated health problems [medquad_ans_0000341-1, medquad_ans_0002230

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.0328) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary ...
2. `medquad_ans_0004627-2` (score=0.0323) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is available, the tabl...
3. `medquad_ans_0003011-1` (score=0.0247) - What are the signs and symptoms of Hyperostosis corticalis generalisata, benign form of Worth with torus palatinus? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperostosis corticalis generalisata, ...
4. `medquad_ans_0003378-1` (score=0.0159) - What are the signs and symptoms of Keratosis palmoplantaris striata 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Keratosis palmoplantaris striata 1. If the information is available, the table belo...
5. `medquad_ans_0000922-1` (score=0.0159) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than normal, particularly the ...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.0318) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyosi...
2. `medquad_ans_0004097-1` (score=0.0306) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polymyositis. Some affect...
3. `medquad_ans_0004097-5` (score=0.0287) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is supported by a blood test...
4. `medquad_ans_0002280-2` (score=0.0275) - What are the signs and symptoms of Familial mixed cryoglobulinemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial mixed cryoglobulinemia. If the information is available, the table below incl...
5. `medquad_ans_0004097-3` (score=0.0164) - What causes mixed connective tissue disease? The exact underlying cause of mixed connective tissue disease (MCTD) is currently unknown. It is an autoimmune disorder, which means the immune system mistakes normal, healthy cells for those ...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Top docs**:

1. `medquad_ans_0000004-5__dup2` (score=0.0318) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate stages of the disease. ...
2. `medquad_ans_0000024-1__dup2` (score=0.0308) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control symptoms or address ...
3. `medquad_ans_0000003-6__dup1` (score=0.0305) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the person will need more and...
4. `medquad_ans_0000153-1__dup2` (score=0.0292) - Summary : Caregivers provide help to another person in need. The person receiving care may be an adult - often a parent or a spouse - or a child with special medical needs. Some caregivers are family members. Others are paid. They do man...
5. `medquad_ans_0000004-23` (score=0.0282) - Currently, no treatment can stop Alzheimer's disease. However, four medications are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time. They work by regulating certain c...

**Answer**:

The provided context does not contain specific information about treatments for Alzheimer's caregiving. It discusses medications for treating symptoms, memory aids, and managing behavioral symptoms, but does not address treatments or strategies specifically for caregivers. Therefore, based on the given context, it is not possible to provide a supported answer regarding treatments for Alzheimer's caregiving. [medquad_ans_0000004-5__dup2]

**Citations detected**: `medquad_ans_0000004-5__dup2`
**Supported citations**: `medquad_ans_0000004-5__dup2`
**Unsupported citations**: _none_

## Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.0325) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identified variations in s...
2. `medquad_ans_0000511-4` (score=0.0309) - Most cases of idiopathic inflammatory myopathy are sporadic, which means they occur in people with no history of the disorder in their family. However, several people with idiopathic inflammatory myopathy have had close relatives with au...
3. `medquad_ans_0000547-3` (score=0.0273) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms result from excessive i...
4. `medquad_ans_0003141-1` (score=0.0267) - Idiopathic inflammatory myopathy refers to a group of conditions that affect the skeletal muscles (muscles used for movement). Although the condition can be diagnosed at any age, idiopathic inflammatory myopathy most commonly occurs in a...
5. `medquad_ans_0000293-3` (score=0.0259) - A mutation in the MATR3 gene has been identified in people with distal myopathy 2. This gene provides instructions for making a protein called matrin 3, which is found in the nucleus of the cell as part of the nuclear matrix. The nuclear...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, particularly within the human leukocyte antigen (HLA) complex. These HLA gene variations are associated with an increased risk of developing the disorder. The HLA complex plays a role in helping the immune system distinguish between the body's own proteins and foreign proteins. Specific variations of HLA genes have been found to affect the risk of developing idiopathic inflammatory myopathy. Researchers are also studying other genes related to immune function to understand their contribution to the disorder's development [medquad_ans_0000511-3]. However, inheriting a genetic variation does not guarantee that a person will develop the condition, as multiple genetic and environmental factors are involved [medquad_ans_0000511-4].

**Citations detected**: `medquad_ans_0000511-3`, `medquad_ans_0000511-4`
**Supported citations**: `medquad_ans_0000511-3`, `medquad_ans_0000511-4`
**Unsupported citations**: _none_

## Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0005223-5` (score=0.0320) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for pyruvate dehydrogen...
2. `medquad_ans_0000855-5` (score=0.0317) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 deficiency - Genetic Te...
3. `medquad_ans_0005223-6` (score=0.0315) - How might pyruvate dehydrogenase deficiency be treated? Treatment of pyruvate dehydrogenase (PDH) deficiency rarely influences the course of the disease, but goals include stimulating the pyruvate dehydrogenase complex (PDHC), providing ...
4. `medquad_ans_0000855-2` (score=0.0305) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
5. `medquad_ans_0005223-1` (score=0.0292) - Pyruvate dehydrogenase deficiency is metabolic disorder associated with abnormal function of the mitochondria in cells, thus depriving the body of energy. Progressive neurological symptoms usually start in infancy but may be evident at b...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.0323) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb abnormalities. All in...
2. `medquad_ans_0006607-2` (score=0.0310) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, the table below includ...
3. `medquad_ans_0006555-2` (score=0.0285) - What are the signs and symptoms of 15q13.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 15q13.3 microdeletion syndrome. If the information is available, the table below includ...
4. `medquad_ans_0006568-1` (score=0.0278) - What are the signs and symptoms of 20p12.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 20p12.3 microdeletion syndrome. If the information is available, the table below includ...
5. `medquad_ans_0001293-1` (score=0.0265) - What are the signs and symptoms of Chromosome 6q25 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chromosome 6q25 microdeletion syndrome. If the information is available, the ta...

**Answer**:

The symptoms of 17q23.1q23.2 microdeletion syndrome include developmental delay, microcephaly, short stature, heart defects, and hand, foot, and limb abnormalities. Specific signs and symptoms reported in the literature include:

- Developmental delay, particularly delays in speech [medquad_ans_0006562-2]
- Microcephaly (seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Short stature (seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Heart defects, including patent ductus arteriosus or atrial septal defects (each seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Limb abnormalities such as long, thin fingers and toes

**Citations detected**: `medquad_ans_0006562-2`
**Supported citations**: `medquad_ans_0006562-2`
**Unsupported citations**: _none_

## Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.0328) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...
2. `medquad_ans_0003860-4` (score=0.0320) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome, one of the two sex...
3. `medquad_ans_0000635-3` (score=0.0311) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
4. `medquad_ans_0000635-4` (score=0.0308) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), a duplication of ...
5. `medquad_ans_0003860-5` (score=0.0306) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the diagnosis.

**Answer**:

MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm of the X chromosome. The MECP2 gene encodes a protein that is important for normal brain functioning and regulates other genes in the brain by switching them on and off. A duplication of the MECP2 gene leads to the production of excess protein, which is unable to properly regulate the expression of other genes, resulting in irregular brain activity and the signs and symptoms of MECP2 duplication syndrome [medquad_ans_0003860-3].

**Citations detected**: `medquad_ans_0003860-3`
**Supported citations**: `medquad_ans_0003860-3`
**Unsupported citations**: _none_

## Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Top docs**:

1. `medquad_ans_0004632-2` (score=0.0323) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the information is available, the ...
2. `medquad_ans_0004633-2` (score=0.0320) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the information is available, the ...
3. `medquad_ans_0004640-2` (score=0.0304) - What are the signs and symptoms of Osteopetrosis autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 7. If the information is available, the table be...
4. `medquad_ans_0004638-2` (score=0.0301) - What are the signs and symptoms of Osteopetrosis autosomal recessive 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 5. If the information is available, the table be...
5. `medquad_ans_0004635-2` (score=0.0294) - What are the signs and symptoms of Osteopetrosis autosomal recessive 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 2. If the information is available, the table be...

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

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Top docs**:

1. `medquad_ans_0003459-1` (score=0.0304) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when antibodies interfer...
2. `medquad_ans_0003459-4` (score=0.0303) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (wher...
3. `medquad_ans_0000084-2__dup1` (score=0.0267) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ephedrine, and 3,4-d...
4. `medquad_ans_0000235-5` (score=0.0233) - These resources address the diagnosis or management of congenital myasthenic syndrome: - Gene Review: Gene Review: Congenital Myasthenic Syndromes - Genetic Testing Registry: CHRNA1-Related Congenital Myasthenic Syndrome - Genetic Testin...
5. `medquad_ans_0003459-3` (score=0.0164) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases where Lambert Eaton ...

**Answer**:

The treatments for Lambert Eaton myasthenic syndrome (LEMS) may include anticholinesterase agents (e.g., pyridostigmine), guanidine hydrochloride, plasmapheresis, IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While some evidence suggests that 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit still needs to be determined. Treatment of the underlying disorder or cancer is the first priority [medquad_ans_0003459-4].

**Citations detected**: `medquad_ans_0003459-4`
**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

## Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.0315) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes how often the sympt...
2. `medquad_ans_0000445-3__dup1` (score=0.0164) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). This may be accomplis...
3. `medquad_ans_0000065-3` (score=0.0161) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This cycle processes exce...
4. `medquad_ans_0000930-2__dup1` (score=0.0161) - What are the signs and symptoms of Carbamoyl phosphate synthetase 1 deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Carbamoyl phosphate synthetase 1 deficiency. If the information is availab...
5. `medquad_ans_0000065-1` (score=0.0159) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxi...

**Answer**:

The signs and symptoms of Arginase deficiency include aminoaciduria (90%), behavioral abnormality (90%), cognitive impairment (90%), neurological speech impairment (90%), EEG abnormality (50%), hemiplegia/hemiparesis (50%), hyperammonemia (50%), seizures (50%), anorexia, autosomal recessive inheritance, diaminoaciduria, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1].

**Citations detected**: `medquad_ans_0000445-2__dup1`
**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

## Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.0325) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but are still small so the...
2. `medquad_ans_0001020-5` (score=0.0309) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau syndrome - Genomics Ed...
3. `medquad_ans_0000961-1__dup1` (score=0.0303) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancreas and, in men, their...
4. `medquad_ans_0006376-1` (score=0.0246) - Von Hippel-Lindau (VHL) disease is an inherited disorder characterized by the abnormal growth of both benign and cancerous tumors and cysts in many parts of the body. Tumors usually first appear in young adulthood. The types of tumors as...
5. `medquad_ans_0001021-5` (score=0.0215) - These resources address the diagnosis or management of von Willebrand disease: - Gene Review: Gene Review: von Willebrand Disease - Genetic Testing Registry: von Willebrand disorder - MedlinePlus Encyclopedia: von Willebrand Disease Thes...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Early detection and treatment are important, and surgical removal of tumors is the primary approach, with radiation therapy used in specific situations [medquad_ans_0000961-1__dup1].

**Citations detected**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Unsupported citations**: _none_

## Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Top docs**:

1. `medquad_ans_0004456-2` (score=0.0318) - What are the signs and symptoms of Noonan syndrome 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 6. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0004453-2` (score=0.0301) - What are the signs and symptoms of Noonan syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 3. If the information is available, the table below includes how often the symptom i...
3. `medquad_ans_0004451-1` (score=0.0294) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
4. `medquad_ans_0004450-1` (score=0.0293) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...
5. `medquad_ans_0004454-1` (score=0.0290) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...

**Answer**:

Noonan syndrome 3 is a genetic disorder characterized by a range of physical and developmental signs and symptoms. According to the Human Phenotype Ontology, signs and symptoms associated with Noonan syndrome 3 include anteverted nares, atrial septal aneurysm, autosomal dominant inheritance, frontal bossing, hypertelorism, juvenile myelomonocytic leukemia, low-set ears, pulmonic stenosis, sagittal craniosynostosis, short nose, short stature, ventricular septal defect, and webbed neck [medquad_ans_0004453-2]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which draws information from sources such as Orphanet, a European rare disease database [medquad_ans_0004453-2].

**Citations detected**: `medquad_ans_0004453-2`
**Supported citations**: `medquad_ans_0004453-2`
**Unsupported citations**: _none_

## Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.0323) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may not have noticeable...
2. `medquad_ans_0001079-1__dup1` (score=0.0320) - What are the signs and symptoms of Cervical ribs, Sprengel anomaly, anal atresia, and urethral obstruction? The Human Phenotype Ontology provides the following list of signs and symptoms for Cervical ribs, Sprengel anomaly, anal atresia,...
3. `medquad_ans_0002966-1` (score=0.0310) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dysplasia, and Sprengel an...
4. `medquad_ans_0001614-1` (score=0.0275) - What are the signs and symptoms of Costocoracoid ligament congenitally short? The Human Phenotype Ontology provides the following list of signs and symptoms for Costocoracoid ligament congenitally short. If the information is available, ...
5. `medquad_ans_0003335-2` (score=0.0257) - What are the signs and symptoms of Juvenile-onset dystonia? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile-onset dystonia. If the information is available, the table below includes how often t...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The condition may also be associated with limb length discrepancy or abnormalities in the spine or ribs [medquad_ans_0005865-2].

**Citations detected**: `medquad_ans_0005865-2`
**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

## Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.0318) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling pathways that are ess...
2. `medquad_ans_0005415-1` (score=0.0318) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant Robinow syndrome. T...
3. `medquad_ans_0005415-3` (score=0.0268) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell must have a mutation...
4. `medquad_ans_0000882-1` (score=0.0264) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the severity of their sig...
5. `medquad_ans_0000882-4` (score=0.0239) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an ind...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome [medquad_ans_0000882-3]. Mutations in the WNT5A or DVL1 gene can cause autosomal dominant Robinow syndrome, with DVL1 mutations specifically associated with the osteosclerotic form [medquad_ans_0000882-3]. In some cases, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause of the condition remains unknown [medquad_ans_0000882-3].

**Citations detected**: `medquad_ans_0000882-3`
**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

## Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Top docs**:

1. `medquad_ans_0000180-1__dup3` (score=0.0305) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a ...
2. `medquad_ans_0000155-12` (score=0.0290) - All of the treatment options for kidney failure require changes and restrictions in your diet. Hemodialysis Hemodialysis has the most restrictions. You should watch how much water and other liquids you get from food and drinks and avoid ...
3. `medquad_ans_0000179-12` (score=0.0290) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should talk with your centers ...
4. `medquad_ans_0000158-13` (score=0.0285) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A persons treatment will dic...
5. `medquad_ans_0000179-8` (score=0.0285) - A kidney transplant places a healthy kidney from another person into your body. The kidney may come from someone who has just died. Your doctor will place your name on a waiting list for a kidney. A family member or friend might be able ...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000180-1__dup3] discusses dietary considerations for kidney failure patients, noting that a special diet is needed to prevent waste buildup in the blood and that hemodialysis has the most restrictions regarding liquids and other intake, but it does not specifically address the topic "Eat Right to Feel Right on Hemodialysis." [doc_id]medquad_ans_0000180-1\_\_dup3[/doc_id]

**Citations detected**: `medquad_ans_0000180-1__dup3`, `doc_id`, `/doc_id`
**Supported citations**: `medquad_ans_0000180-1__dup3`
**Unsupported citations**: `doc_id`, `/doc_id`

## Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.0325) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000677-1` (score=0.0259) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild form is called atypic...
3. `medquad_ans_0000359-2` (score=0.0161) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0000677-3` (score=0.0159) - Mutations in the MCOLN1 gene cause mucolipidosis type IV. This gene provides instructions for making a protein called mucolipin-1. This protein is located in the membranes of lysosomes and endosomes, compartments within the cell that dig...
5. `medquad_ans_0000679-2` (score=0.0159) - MPS II occurs in approximately 1 in 100,000 to 1 in 170,000 males.

**Answer**:

Mucolipidosis type IV affects approximately 1 in 40,000 people [medquad_ans_0000677-2]. Another source estimates the condition affects about 1 per million people worldwide [medquad_ans_0000359-2]. These figures may reflect different population estimates or regional variations.

**Citations detected**: `medquad_ans_0000677-2`, `medquad_ans_0000359-2`
**Supported citations**: `medquad_ans_0000677-2`, `medquad_ans_0000359-2`
**Unsupported citations**: _none_

## Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.0328) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin con...
2. `medquad_ans_0001913-1` (score=0.0289) - Dowling-Degos disease is a skin condition characterized by a lacy or net-like (reticulate) pattern of abnormally dark skin coloring (hyperpigmentation), particularly in the body's folds and creases. Other features may include dark lesion...
3. `medquad_ans_0001913-2` (score=0.0267) - What are the signs and symptoms of Dowling-Degos disease? Dowling-Degos disease is characterized by a lacy or net-like (reticulate) pattern of abnormally dark skin coloring (hyperpigmentation), particularly in the body's folds and crease...
4. `medquad_ans_0000718-1__dup2` (score=0.0255) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts or reshape sagging ...
5. `medquad_ans_0000302-1` (score=0.0253) - Dowling-Degos disease is a skin condition characterized by a lacy or net-like (reticulate) pattern of abnormally dark skin coloring (hyperpigmentation), particularly in the body's folds and creases. These skin changes typically first app...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Citations detected**: `medquad_ans_0000795-1__dup1`
**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

## Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Top docs**:

1. `medquad_ans_0001045-1` (score=0.0325) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, the fatty covering (...
2. `medquad_ans_0001045-4` (score=0.0323) - X-linked adrenoleukodystrophy is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males (who...
3. `medquad_ans_0001045-3` (score=0.0310) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules called very long-chain f...
4. `medquad_ans_0001045-5` (score=0.0161) - These resources address the diagnosis or management of X-linked adrenoleukodystrophy: - Gene Review: Gene Review: X-Linked Adrenoleukodystrophy - Genetic Testing Registry: Adrenoleukodystrophy - Genomics Education Programme (UK) - Medlin...
5. `medquad_ans_0001058-4` (score=0.0159) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males (who have only one ...

**Answer**:

The provided context does not specify the exact genetic changes related to X-linked adrenoleukodystrophy. It describes the disorder's effects on the nervous system and adrenal glands, as well as its types and symptoms, but does not detail the underlying genetic mutations or molecular mechanisms. [medquad_ans_0001045-1]

**Citations detected**: `medquad_ans_0001045-1`
**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

## Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.0325) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of European descent, where it...
2. `medquad_ans_0000988-4` (score=0.0248) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ra...
3. `medquad_ans_0000988-1` (score=0.0164) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most frequently occur in the...
4. `medquad_ans_0000650-2` (score=0.0161) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
5. `medquad_ans_0000988-5` (score=0.0159) - These resources address the diagnosis or management of transthyretin amyloidosis: - Boston University: Amyloid Treatment & Research Program - Gene Review: Gene Review: Familial Transthyretin Amyloidosis - Genetic Testing Registry: Amyloi...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Citations detected**: `medquad_ans_0000988-2`
**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

## Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.0328) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the information is available, the ...
2. `medquad_ans_0006016-1` (score=0.0302) - What are the signs and symptoms of Taurodontism, microdontia, and dens invaginatus? The Human Phenotype Ontology provides the following list of signs and symptoms for Taurodontism, microdontia, and dens invaginatus. If the information is...
3. `medquad_ans_0006174-1` (score=0.0161) - What are the signs and symptoms of Tricho-dento-osseous syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome. If the information is available, the table below includes...
4. `medquad_ans_0005769-1` (score=0.0159) - What are the signs and symptoms of Spinocerebellar ataxia 20? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia 20. If the information is available, the table below includes how oft...
5. `medquad_ans_0006175-1` (score=0.0159) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, the table below incl...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include those listed by the Human Phenotype Ontology (HPO), which compiles information on how often a sign or symptom occurs in individuals with this condition. The frequency of these symptoms is typically estimated as a percentage or presented as a fraction (e.g., 25/25), indicating the number of patients with the symptom out of the total number examined in a specific study. However, the specific symptoms themselves are not detailed in the provided context [medquad_ans_0001780-1].

**Citations detected**: `medquad_ans_0001780-1`
**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

## Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.0311) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0000782-1` (score=0.0256) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia means "out of place." I...
3. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0004818-2` (score=0.0161) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discovered when magnetic re...
5. `medquad_ans_0004818-1` (score=0.0159) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected people typically develo...

**Answer**:

Periventricular heterotopia affects about 1 per million people worldwide [medquad_ans_0000359-2].

**Citations detected**: `medquad_ans_0000359-2`
**Supported citations**: `medquad_ans_0000359-2`
**Unsupported citations**: _none_

## Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Top docs**:

1. `medquad_ans_0000007_2-6` (score=0.0164) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
2. `medquad_ans_0000169-4` (score=0.0164) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.
3. `medquad_ans_0000009_2-8` (score=0.0161) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ongoing clinical tri...
4. `medquad_ans_0000179-4__dup1` (score=0.0161) - The NINDS conducts and supports a broad program of basic and clinical research on epilepsy including Lennox-Gastaut syndrome. These studies are aimed at finding the causes of these disorders, improving the diagnosis, and developing new m...
5. `medquad_ans_0000032_1-8` (score=0.0159) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

Answer: The provided context does not mention any research or clinical trials being conducted for Landau-Kleffner Syndrome. [medquad_ans_0000007_2-6], [medquad_ans_0000169-4], and [medquad_ans_0000009_2-8] discuss clinical trials for cancer treatments, sleep disorders, and general cancer therapies, but none reference Landau-Kleffner Syndrome. Therefore, no relevant information is available in the context.

**Citations detected**: `medquad_ans_0000007_2-6`, `medquad_ans_0000169-4`, `medquad_ans_0000009_2-8`
**Supported citations**: `medquad_ans_0000007_2-6`, `medquad_ans_0000169-4`, `medquad_ans_0000009_2-8`
**Unsupported citations**: _none_

## Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Top docs**:

1. `medquad_ans_0001935-1` (score=0.0298) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystrophin. Boys with Duchen...
2. `medquad_ans_0000576-1` (score=0.0294) - LAMA2-related muscular dystrophy is a disorder that causes weakness and wasting (atrophy) of muscles used for movement (skeletal muscles). This condition generally appears in one of two ways: as a severe, early-onset type or a milder, la...
3. `medquad_ans_0000334-1` (score=0.0270) - Facioscapulohumeral muscular dystrophy is a disorder characterized by muscle weakness and wasting (atrophy). This condition gets its name from the muscles that are affected most often: those of the face (facio-), around the shoulder blad...
4. `medquad_ans_0001935-5` (score=0.0263) - How might Duchenne muscular dystrophy be treated? There is no known cure for Duchenne muscular dystrophy (DMD). Treatment is aimed at the control of symptoms to maximize the quality of life. Individuals with DMD often experience dilated ...
5. `medquad_ans_0000602-3` (score=0.0259) - The various forms of limb-girdle muscular dystrophy are caused by mutations in many different genes. These genes provide instructions for making proteins that are involved in muscle maintenance and repair. Some of the proteins produced f...

**Answer**:

Muscular dystrophy is a group of disorders characterized by progressive weakness and wasting (atrophy) of muscles used for movement (skeletal muscles). Duchenne muscular dystrophy (DMD) is a rapidly progressive form that primarily affects boys and is caused by a mutation in the DMD gene, leading to the absence of the dystrophin protein in muscles. DMD results in progressive loss of muscle function and weakness, beginning in the lower limbs, and may also affect the heart. It is inherited in an X-linked recessive pattern and has no known cure; treatment focuses on symptom control to maximize quality of life [medquad_ans_0001935-1].

LAMA2-related muscular dystrophy is another type that causes muscle weakness and atrophy and can present in two forms: early-onset (appearing at birth or within the first few months, often classified as congenital muscular dystrophy type 1A) and late

**Citations detected**: `medquad_ans_0001935-1`
**Supported citations**: `medquad_ans_0001935-1`
**Unsupported citations**: _none_

## Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Top docs**:

1. `medquad_ans_0005020-1` (score=0.0320) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the table below includes how...
2. `medquad_ans_0005022-1` (score=0.0303) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the table below includes how...
3. `medquad_ans_0004948-1` (score=0.0297) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no oth...
4. `medquad_ans_0005971-1` (score=0.0294) - What are the signs and symptoms of Syndactyly-polydactyly-earlobe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly-polydactyly-earlobe syndrome. If the information is available, the ...
5. `medquad_ans_0005019-1` (score=0.0285) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the table below includes how...

**Answer**:

Polydactyly is a congenital condition characterized by the presence of extra fingers or toes. In the context of Preaxial polydactyly type 2, it involves duplication of the thumb phalanx, preaxial hand and foot polydactyly, and triphalangeal thumbs [medquad_ans_0005020-1]. In Preaxial polydactyly type 4, it includes preaxial polydactyly and dysplastic distal thumb phalanges with a central hole [medquad_ans_0005022-1]. These conditions are typically associated with autosomal dominant inheritance [medquad_ans_0005020-1][medquad_ans_0005022-1].

**Citations detected**: `medquad_ans_0005020-1`, `medquad_ans_0005022-1`
**Supported citations**: `medquad_ans_0005020-1`, `medquad_ans_0005022-1`
**Unsupported citations**: _none_

## Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Top docs**:

1. `medquad_ans_0002519-1` (score=0.0164) - Hereditary geniospasm is a movement disorder that causes episodes of involuntary tremors of the chin and lower lip. The episodes may last anywhere from a few seconds to hours and may occur spontaneously or be brought on by stress. The ep...
2. `medquad_ans_0002519-2` (score=0.0164) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the symptom is seen in peop...
3. `medquad_ans_0002519-3` (score=0.0161) - How is hereditary geniospasm inherited? Hereditary geniospasm is inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene in each body cell is sufficient to cause signs and symptoms of...
4. `medquad_ans_0001057-1__dup1` (score=0.0161) - What are the signs and symptoms of Cerebral gigantism jaw cysts? The Human Phenotype Ontology provides the following list of signs and symptoms for Cerebral gigantism jaw cysts. If the information is available, the table below includes h...
5. `medquad_ans_0000108-2__dup1` (score=0.0159) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...

**Answer**:

The symptoms of Geniospasm include episodes of involuntary tremors of the chin and lower lip. These episodes may last from a few seconds to hours and can occur spontaneously or be triggered by stress. The condition typically first appears in infancy or childhood and tends to decrease in frequency with age [medquad_ans_0002519-1].

**Citations detected**: `medquad_ans_0002519-1`
**Supported citations**: `medquad_ans_0002519-1`
**Unsupported citations**: _none_

## Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000512-3` (score=0.0325) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of developing idiopathic pul...
2. `medquad_ans_0000512-4` (score=0.0320) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance. Autosomal dominant ...
3. `medquad_ans_0000078-4__dup2` (score=0.0294) - Sometimes doctors can find out what is causing pulmonary fibrosis (lung scarring). For example, exposure to environmental pollutants and certain medicines can cause the disease. Environmental pollutants include inorganic dust (silica and...
4. `medquad_ans_0000512-1` (score=0.0290) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effectively. The disease u...
5. `medquad_ans_0000758-1__dup2` (score=0.0284) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in the TERC and TERT genes, which have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. However, it is not well understood how defects in telomerase are associated with the lung damage characteristic of idiopathic pulmonary fibrosis. Most of the identified genetic changes account for only a small proportion of cases [medquad_ans_0000512-3].

**Citations detected**: `medquad_ans_0000512-3`
**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

## Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Top docs**:

1. `medquad_ans_0005008-1` (score=0.0164) - Potocki-Shaffer syndrome is a contiguous gene deletion syndrome associated with deletions in a specific region of chromosome 11 (11p11.2). The characteristic features of Potocki-Shaffer syndrome include openings in the two bones that for...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000813-3` (score=0.0161) - Potocki-Shaffer syndrome (also known as proximal 11p deletion syndrome) is caused by a deletion of genetic material from the short (p) arm of chromosome 11 at a position designated 11p11.2. The size of the deletion varies among affected ...
4. `medquad_ans_0000884-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000813-2` (score=0.0159) - Potocki-Shaffer syndrome is a rare condition, although its prevalence is unknown. Fewer than 100 cases have been reported in the scientific literature.

**Answer**:

The provided context does not indicate that Potocki-Shaffer syndrome is inherited. It describes Potocki-Shaffer syndrome as a contiguous gene deletion syndrome caused by deletions in the 11p11.2 region of chromosome 11, with features resulting from the loss of specific genes such as ALX4 and EXT2. While one document mentions an autosomal recessive pattern, this refers to a different condition, not Potocki-Shaffer syndrome [medquad_ans_0000785-4]. There is no explicit statement in the context confirming that Potocki-Shaffer syndrome is inherited. Therefore, based on the available information, it cannot be determined that Potocki-Shaffer syndrome is inherited.

Answer: No, Potocki-Shaffer syndrome is not stated to be inherited in the provided context. [medquad_ans_0005008-1]

**Citations detected**: `medquad_ans_0000785-4`, `medquad_ans_0005008-1`
**Supported citations**: `medquad_ans_0000785-4`, `medquad_ans_0005008-1`
**Unsupported citations**: _none_

## Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Top docs**:

1. `medquad_ans_0000925-1__dup1` (score=0.0320) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about your baby's health, ca...
2. `medquad_ans_0000223-10` (score=0.0286) - You can take the following steps to help prevent your child from getting a UTI: - Teach your child not to hold in urine and to go to the bathroom whenever your child feels the urge. - Teach your child how to properly clean himself or her...
3. `medquad_ans_0000131-1__dup3` (score=0.0243) - Summary : Breastfeeding offers many benefits to your baby. Breast milk contains the right balance of nutrients to help your infant grow into a strong and healthy toddler. Some of the nutrients in breast milk also help protect your infant...
4. `medquad_ans_0000223-9` (score=0.0241) - Talk with your childs health care provider after your childs UTI is gone. The health care provider may want to do more tests to check for VUR or a blockage in the urinary tract. Repeated infections in an abnormal urinary tract may cause ...
5. `medquad_ans_0000010-5__dup1` (score=0.0161) - Your primary care doctor will diagnose asthma based on your medical and family histories, a physical exam, and test results. Your doctor also will figure out the severity of your asthmathat is, whether it's intermittent, mild, moderate, ...

**Answer**:

No, there is no information about child care in the provided context. [medquad_ans_0000925-1__dup1, medquad_ans_0000223-10, medquad_ans_0000131-1__dup3, medquad_ans_0000223-9] discuss topics such as infant health, urinary tract infections, breastfeeding, and when to consult a health care provider, but do not address child care.

**Citations detected**: `medquad_ans_0000925-1__dup1, medquad_ans_0000223-10, medquad_ans_0000131-1__dup3, medquad_ans_0000223-9`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000925-1__dup1, medquad_ans_0000223-10, medquad_ans_0000131-1__dup3, medquad_ans_0000223-9`

## Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.0320) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000046-4` (score=0.0282) - Alternating hemiplegia of childhood is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of alternating hemiplegia of childhood result from n...
3. `medquad_ans_0000046-1` (score=0.0257) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis alternates from one side o...
4. `medquad_ans_0000016-3__dup1` (score=0.0252) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy, and balance and gai...
5. `medquad_ans_0000016-1__dup1` (score=0.0247) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or both sides of the body,...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Citations detected**: `medquad_ans_0000046-2`
**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

## Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Top docs**:

1. `medquad_ans_0000159-3__dup2` (score=0.0290) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time or shortly after lu...
2. `medquad_ans_0003687-2` (score=0.0265) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype Ontology provides th...
3. `medquad_ans_0001448-2` (score=0.0229) - What are the signs and symptoms of Complement component 2 deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Complement component 2 deficiency. If the information is available, the table below ...
4. `medquad_ans_0000887-2__dup1` (score=0.0209) - What are the signs and symptoms of C1q deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for C1q deficiency. If the information is available, the table below includes how often the symptom is seen...
5. `medquad_ans_0003687-1` (score=0.0164) - Lupus is an autoimmune disease that can affect almost every organ in the body. Symptoms of lupus can range from very mild to life-threatening. There are three types of lupus; systemic lupus erythematosus, discoid lupus, and drug-induced ...

**Answer**:

The symptoms of lupus include joint pain or swelling, muscle pain, fever with no known cause, red rashes often on the face (called butterfly rashes due to their shape), abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, edema of the lower limbs, glomerulopathy, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood, increased intracranial pressure, lymphadenopathy, lymphopenia, meningitis, myalgia, normocytic anemia, recurrent respiratory infections, renal insufficiency, sleep disturbance, splenomegaly, and weight loss [medquad*ans*

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Top docs**:

1. `medquad_ans_0001395-1` (score=0.0313) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
2. `medquad_ans_0001398-2` (score=0.0313) - What are the signs and symptoms of Cockayne syndrome type III? The Human Phenotype Ontology provides the following list of signs and symptoms for Cockayne syndrome type III. If the information is available, the table below includes how o...
3. `medquad_ans_0001397-1` (score=0.0313) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
4. `medquad_ans_0001398-1` (score=0.0313) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
5. `medquad_ans_0000194-3` (score=0.0308) - Cockayne syndrome can result from mutations in either the ERCC6 gene (also known as the CSB gene) or the ERCC8 gene (also known as the CSA gene). These genes provide instructions for making proteins that are involved in repairing damaged...

**Answer**:

Cockayne syndrome is a rare condition that causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. It also includes failure to thrive in the newborn, microcephaly, and impaired nervous system development. Other symptoms may include hearing loss, tooth decay, and eye and bone abnormalities. Cockayne syndrome is caused by mutations in either the ERCC8 (CSA) or ERCC6 (CSB) genes and is inherited in an autosomal recessive pattern. The syndrome is classified into three types: type 1 (classic or "moderate"), type 2 (severe or "early-onset"), and type 3 (milder). Type 1 is diagnosed during early childhood, type 2 presents with growth and developmental abnormalities at birth, and type 3 is a milder form. The typical lifespan for type 1 is ten to twenty years, individuals with type 2 usually

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Top docs**:

1. `medquad_ans_0000029-1__dup5` (score=0.0313) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is reversible with goo...
2. `medquad_ans_0000029-8__dup1` (score=0.0310) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regular cleaning by a de...
3. `medquad_ans_0000029-6__dup1` (score=0.0310) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead to sore or bleeding...
4. `medquad_ans_0000029-3__dup2` (score=0.0306) - Symptoms Symptoms of gum disease may include: - bad breath that won't go away - red or swollen gums - tender or bleeding gums - painful chewing - loose teeth - sensitive teeth - receding gums or longer appearing teeth bad breath that won...
5. `medquad_ans_0000029-16` (score=0.0304) - People are not often aware they have gum disease until it is advanced. Any of these symptoms may be a sign of a serious problem and should be checked by a dentist. - bad breath that won't go away - red or swollen gums - tender or bleedin...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is reversible with good oral hygiene, and periodontitis, a more severe form that can damage the soft tissues and bone that support teeth. If left untreated, periodontitis can lead to tooth loss [medquad_ans_0000029-1__dup5]. In gingivitis, the gums become red, swollen, and can bleed easily, and this form does not include any loss of bone or tissue that hold teeth in place. When gingivitis is not treated, it can advance to periodontitis, where gums pull away from the teeth and form infected pockets below the gum line, leading to breakdown of bone and connective tissue that support the teeth [medquad_ans_0000029-1\_\_dup

**Citations detected**: `medquad_ans_0000029-1__dup5`
**Supported citations**: `medquad_ans_0000029-1__dup5`
**Unsupported citations**: _none_

## Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.0320) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000505-2` (score=0.0283) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
3. `medquad_ans_0003059-4` (score=0.0223) - How is hypokalemic periodic paralysis inherited? This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person h...
4. `medquad_ans_0000494-5` (score=0.0164) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis - Genetic Testing Re...
5. `medquad_ans_0000494-1` (score=0.0161) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Citations detected**: `medquad_ans_0000494-2`
**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

## Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Top docs**:

1. `medquad_ans_0005212-3` (score=0.0325) - Is Pyle disease inherited? Pyle disease in inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one cop...
2. `medquad_ans_0005212-2` (score=0.0323) - What are the signs and symptoms of Pyle disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyle disease. If the information is available, the table below includes how often the symptom is seen in ...
3. `medquad_ans_0005212-1` (score=0.0320) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widenin...
4. `medquad_ans_0000115-8__dup1` (score=0.0223) - - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall. - Scientists are ...
5. `medquad_ans_0001853-3` (score=0.0156) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widening of the ribs and clavicles (collarbones), platyspondyly (flattening of the bones of the spine), and cortical thinning. It is inherited in an autosomal recessive manner, meaning both copies of the gene in each cell have mutations. The parents of an individual with Pyle disease each carry one copy of the mutated gene but typically do not have signs or symptoms of the condition. Some carriers (obligate heterozygotes) may show minor skeletal changes [medquad_ans_0005212-1][medquad_ans_0005212-3]. Approximately 30 cases have been reported in

**Citations detected**: `medquad_ans_0005212-1`, `medquad_ans_0005212-3`
**Supported citations**: `medquad_ans_0005212-1`, `medquad_ans_0005212-3`
**Unsupported citations**: _none_

## Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Top docs**:

1. `medquad_ans_0000256-1__dup3` (score=0.0325) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thrombosis can break loo...
2. `medquad_ans_0000051-4__dup2` (score=0.0318) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden) will do this. Horm...
3. `medquad_ans_0000104-1__dup4` (score=0.0313) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent blood clots from diss...
4. `medquad_ans_0000051-5__dup1` (score=0.0307) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause serious, possibly lif...
5. `medquad_ans_0000051-1__dup2` (score=0.0296) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg or thigh. They also c...

**Answer**:

Deep vein thrombosis (DVT) is caused by conditions or factors that make the blood thicker or more likely to clot than normal. Risk factors include certain genetic disorders (such as factor V Leiden), hormone therapy or birth control pills, injury to a deep vein from surgery, broken bones, or trauma, slow blood flow due to lack of movement (e.g., after surgery, being ill and in bed for a long time, or prolonged travel), pregnancy and the first 6 weeks after giving birth, recent or ongoing cancer treatment, presence of a central venous catheter, older age (over 60), overweight or obesity, and smoking [medquad_ans_0000051-4__dup2]. Additionally, conditions such as atherosclerosis, diabetes, atrial fibrillation, metabolic syndrome, and certain medicines can contribute to excessive blood clotting, which may lead to DVT [medquad_ans_0

**Citations detected**: `medquad_ans_0000051-4__dup2`
**Supported citations**: `medquad_ans_0000051-4__dup2`
**Unsupported citations**: _none_

## Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Top docs**:

1. `medquad_ans_0000178-3__dup2` (score=0.0315) - The prognosis for individuals with Leigh's disease is poor. Individuals who lack mitochondrial complex IV activity and those with pyruvate dehydrogenase deficiency tend to have the worst prognosis and die within a few years. Those with p...
2. `medquad_ans_0000661-2` (score=0.0312) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.
3. `medquad_ans_0004079-4` (score=0.0311) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Foundation (UMDF) provides...
4. `medquad_ans_0000661-5` (score=0.0311) - These resources address the diagnosis or management of mitochondrial complex III deficiency: - Gene Review: Gene Review: Mitochondrial Disorders Overview - Genetic Testing Registry: MITOCHONDRIAL COMPLEX III DEFICIENCY, NUCLEAR TYPE 6 - ...
5. `medquad_ans_0000661-3` (score=0.0301) - Mitochondrial complex III deficiency can be caused by mutations in one of several genes. The proteins produced from these genes either are a part of or help assemble a group of proteins called complex III. The two most commonly mutated g...

**Answer**:

The provided context does not contain information about mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given context. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.0325) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most affected people also ...
2. `medquad_ans_0000496-1__dup1` (score=0.0320) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is available, the tabl...
3. `medquad_ans_0000078-2` (score=0.0317) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
4. `medquad_ans_0000078-5` (score=0.0295) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 2 - Genetic Testing ...
5. `medquad_ans_0005768-2` (score=0.0294) - What are the signs and symptoms of Spinocerebellar ataxia 2? Early symptoms of spinocerebellar ataxia may include uncoordinated movement (ataxia) and leg cramps. Other symptoms may include tremor; decreased muscle tone; poor tendon refle...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see things in their peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4; in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Top docs**:

1. `medquad_ans_0001744-1` (score=0.0323) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine aplasia microtia and mic...
2. `medquad_ans_0000221-1` (score=0.0320) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that form the inner ear are...
3. `medquad_ans_0000221-5` (score=0.0318) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with labyrinthine aplasia mic...
4. `medquad_ans_0000012-1__dup1` (score=0.0253) - What are the signs and symptoms of Ablepharon macrostomia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ablepharon macrostomia syndrome. If the information is available, the table below incl...
5. `medquad_ans_0000248-1` (score=0.0161) - Craniofacial microsomia is a term used to describe a spectrum of abnormalities that primarily affect the development of the skull (cranium) and face before birth. Microsomia means abnormal smallness of body structures. Most people with c...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed [medquad_ans_0000221-1].

**Citations detected**: `medquad_ans_0000221-1`
**Supported citations**: `medquad_ans_0000221-1`
**Unsupported citations**: _none_

## Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.0164) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in the gene, and occur ...
2. `medquad_ans_0000441-4` (score=0.0164) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family members with the con...
3. `medquad_ans_0000420-3__dup1` (score=0.0161) - How is Apert syndrome diagnosed? Is genetic testing needed to confirm the diagnosis? Apert syndrome and the other conditions associated with FGFR-related craniosynostosis were clinically defined long before the molecular basis of this gr...
4. `medquad_ans_0000785-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000064-1` (score=0.0159) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and face. In addition, a v...

**Answer**:

Yes, Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases result from new mutations and occur in individuals with no family history of the disorder, although affected individuals can pass the condition to their offspring [medquad_ans_0000064-4]. Clinical genetic testing, such as FGFR2 sequence analysis, can confirm the diagnosis and is highly sensitive, detecting over 98% of cases due to mutations in the 7th exon of the FGFR2 gene [medquad_ans_0000420-3__dup1].

**Citations detected**: `medquad_ans_0000064-4`, `medquad_ans_0000420-3__dup1`
**Supported citations**: `medquad_ans_0000064-4`, `medquad_ans_0000420-3__dup1`
**Unsupported citations**: _none_

## Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.0323) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organisms are excreted in ...
2. `medquad_ans_0000341-1__dup3` (score=0.0323) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella burnetii usually devel...
3. `medquad_ans_0000341-3__dup2` (score=0.0315) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact the Rickettsial Zoon...
4. `medquad_ans_0000341-4__dup2` (score=0.0315) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities housing sheep. Prev...
5. `medquad_ans_0000359-1__dup1` (score=0.0258) - A fever is a body temperature that is higher than normal. It is not an illness. It is part of your body's defense against infection. Most bacteria and viruses that cause infections do well at the body's normal temperature (98.6 F). A sli...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. Treatment of the acute form involves antibiotics, while treatment for the chronic form depends on the symptoms

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=0.0328) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interstitial lung disease o...
2. `medquad_ans_0000038-9` (score=0.0320) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A counselor can explai...
3. `medquad_ans_0001152-2` (score=0.0318) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment for chILD is to rel...
4. `medquad_ans_0000038-1__dup2` (score=0.0302) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chronic cough, rapid breat...
5. `medquad_ans_0001152-1` (score=0.0294) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD decrease a child's ab...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD), those with an inherited surfactant disorder or a family history of such disorders, children with problems involving aspiration (inhaling food, liquid, or vomit into the lungs), children with immune system disorders, children exposed to environmental lung irritants such as molds and chemicals, children with systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), children undergoing cancer treatments like radiation or chemotherapy, and children who have had a bone marrow or lung transplant are at increased risk for childhood interstitial lung disease [medquad_ans_0000038-5__dup1]. Additionally, certain types of chILD are more common in infants and young children, while others can occur in children of any age [medquad_ans_0000038-5__dup1].

**Citations detected**: `medquad_ans_0000038-5__dup1`
**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

## Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.0328) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in late childhood or early...
2. `medquad_ans_0000404-1__dup1` (score=0.0320) - Giant cell arteritis is a disorder that causes inflammation of your arteries, usually in the scalp, neck, and arms. It narrows the arteries, which keeps blood from flowing well. Giant cell arteritis often occurs with another disorder cal...
3. `medquad_ans_0000087-1__dup1` (score=0.0283) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused by other immune sys...
4. `medquad_ans_0005164-1` (score=0.0260) - Psoriatic juvenile idiopathic arthritis is a subtype of juvenile idiopathic arthritis that is characterized by both arthritis and psoriasis. Other signs and symptoms may include dactylitis (inflammation and swelling of an entire finger o...
5. `medquad_ans_0006001-1` (score=0.0258) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the inflammation, the blood v...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels, specifically affecting the temporal arteries located at the lower sides of the skull, directly underneath the temple. It is generally diagnosed in late childhood or early adulthood and often presents with no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is usually treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Citations detected**: `medquad_ans_0003334-1`
**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

## Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0001082-2` (score=0.0328) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyryl-CoA dehydrogenase ...
2. `medquad_ans_0001082-1` (score=0.0320) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as organic acids. Abno...
3. `medquad_ans_0006577-1` (score=0.0315) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particular amino acid called...
4. `medquad_ans_0001082-5` (score=0.0315) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from MedlinePlus offer i...
5. `medquad_ans_0001082-3` (score=0.0308) - Mutations in the ACADSB gene cause 2-methylbutyryl-CoA dehydrogenase deficiency. The ACADSB gene provides instructions for making an enzyme called 2-methylbutyryl-CoA dehydrogenase that helps process the amino acid isoleucine. Mutations ...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the abnormal buildup of organic acids in the blood, urine, and tissues, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy, vomiting, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills. Symptoms can be triggered by fasting, infections, or increased protein intake. The disorder is more common among individuals of Hmong ancestry, with an incidence of 1 in 250 to 1 in 500 people of Hmong ancestry [

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.0307) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information is available, the ta...
2. `medquad_ans_0005845-1` (score=0.0298) - What are the signs and symptoms of Spondyloepiphyseal dysplasia-brachydactyly and distinctive speech? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia-brachydactyly and disti...
3. `medquad_ans_0005839-1` (score=0.0290) - What are the signs and symptoms of Spondyloepiphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia. If the information is available, the table below includes h...
4. `medquad_ans_0004193-1` (score=0.0236) - What are the signs and symptoms of Multiple epiphyseal dysplasia 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple epiphyseal dysplasia 4. If the information is available, the table below incl...
5. `medquad_ans_0002946-1` (score=0.0224) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Top docs**:

1. `medquad_ans_0000853-2` (score=0.0320) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
2. `medquad_ans_0000853-1` (score=0.0313) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures...
3. `medquad_ans_0005219-3` (score=0.0308) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations....
4. `medquad_ans_0005219-4` (score=0.0308) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of seizure are medically ...
5. `medquad_ans_0005219-1` (score=0.0305) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures...

**Answer**:

Pyridoxine-dependent epilepsy is caused by mutations in the ALDH7A1 gene, which lead to a deficiency of the enzyme α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the amino acid lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. The exact mechanism by which a lack of pyridoxine causes seizures is unclear. In some individuals, no mutations in the ALDH7A1 gene are identified, and the cause of the condition remains unknown [medquad_ans_0005219-3].

**Citations detected**: `medquad_ans_0005219-3`
**Supported citations**: `medquad_ans_0005219-3`
**Unsupported citations**: _none_

## Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.0323) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk f...
2. `medquad_ans_0000004_1-2` (score=0.0315) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not havin...
3. `medquad_ans_0000004_6-6` (score=0.0303) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spreads in the body. - Sta...
4. `medquad_ans_0000004_7-1` (score=0.0296) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major types of childhood n...
5. `medquad_ans_0000004_6-5` (score=0.0296) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whether there are B symp...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Citations detected**: `medquad_ans_0000004_6-2`
**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

## Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Top docs**:

1. `medquad_ans_0001795-2` (score=0.0320) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table below includes how oft...
2. `medquad_ans_0000095-1__dup4` (score=0.0318) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation may precede lesion f...
3. `medquad_ans_0001795-1` (score=0.0318) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow onset of symptoms u...
4. `medquad_ans_0001795-3` (score=0.0317) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, dapsone may cause ser...
5. `medquad_ans_0000095-2__dup3` (score=0.0305) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immune response to gluten...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, which are more common on the knees, elbows, buttocks, and shoulder blades [medquad_ans_0001795-1]. It is often associated with celiac disease and may present with small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp, with the face and groin also potentially involved [medquad_ans_0000095-1__dup4]. A burning sensation may precede lesion formation, and lesions are often scratched off, appearing as erosions and excoriations [medquad_ans_0000095-1__dup4]. Patients may also experience dental enamel defects in permanent teeth, which is another manifestation of celiac disease [medquad_ans_000

**Citations detected**: `medquad_ans_0001795-1`, `medquad_ans_0000095-1__dup4`
**Supported citations**: `medquad_ans_0001795-1`, `medquad_ans_0000095-1__dup4`
**Unsupported citations**: _none_

## Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.0320) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Registry: Congenital myo...
2. `medquad_ans_0000040-2__dup1` (score=0.0301) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some succes...
3. `medquad_ans_0000084-2__dup1` (score=0.0199) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ephedrine, and 3,4-d...
4. `medquad_ans_0004287-2` (score=0.0164) - What are the signs and symptoms of Myotonia congenita autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Myotonia congenita autosomal dominant. If the information is available, the tabl...
5. `medquad_ans_0000040-4__dup1` (score=0.0161) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the country. Current research ...

**Answer**:

Treatments for myotonia congenita include physical therapy and other rehabilitative therapies to improve muscle function. For individuals with more limiting symptoms, medications such as quinine or anticonvulsant drugs like phenytoin have been used with some success [medquad_ans_0000040-2__dup1]. Most people with myotonia congenita do not require special treatments, as stiff muscles usually resolve with exercise or light movement, especially after rest [medquad_ans_0000040-2__dup1].

Note: The provided context does not specify treatments for the autosomal dominant or recessive subtypes of myotonia congenita in detail, and treatments for congenital myasthenia (a different condition) are not directly applicable to myotonia congenita [medquad_ans_0000084-2__dup1].

**Citations detected**: `medquad_ans_0000040-2__dup1`, `medquad_ans_0000084-2__dup1`
**Supported citations**: `medquad_ans_0000040-2__dup1`, `medquad_ans_0000084-2__dup1`
**Unsupported citations**: _none_

## Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Top docs**:

1. `medquad_ans_0006449-5` (score=0.0318) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, whi...
2. `medquad_ans_0000133-11` (score=0.0308) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the initial treatment and...
3. `medquad_ans_0000133-13` (score=0.0307) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correctly and copper build...
4. `medquad_ans_0000973-1__dup1` (score=0.0306) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver releases extra copper i...
5. `medquad_ans_0000133-10` (score=0.0304) - A health care provider will treat Wilson disease with a lifelong effort to reduce and control the amount of copper in the body. Treatment may include - medications - changes in eating, diet, and nutrition - a liver transplant Medications...

**Answer**:

Treatment for Wilson disease involves lifelong management to reduce or control copper accumulation in the body. This may include medications and dietary modifications, such as avoiding foods high in copper (e.g., shellfish, liver, mushrooms, nuts, chocolate) [medquad_ans_0006449-5]. Individuals should also check their tap water for copper levels, especially if it runs through copper pipes or comes from a well, and avoid using copper containers or cookware for storing or preparing food or drinks [medquad_ans_0000133-11]. Patients should discuss the use of vitamins and dietary supplements with their healthcare provider, as some supplements contain copper and may be harmful; a pharmacist can recommend copper-free options [medquad_ans_0000133-11]. If treatment is not effective or if liver failure develops—such as due to cirrhosis or acute liver failure—a liver transplant may be

**Citations detected**: `medquad_ans_0006449-5`, `medquad_ans_0000133-11`
**Supported citations**: `medquad_ans_0006449-5`, `medquad_ans_0000133-11`
**Unsupported citations**: _none_

## Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.0280) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are related to an existing...
2. `medquad_ans_0000814-3` (score=0.0164) - Mutations in the MECP2 gene cause PPM-X syndrome. The MECP2 gene provides instructions for making a protein called MeCP2 that is critical for normal brain function. Researchers believe that this protein has several functions, including r...
3. `medquad_ans_0000592-1` (score=0.0161) - Lennox-Gastaut syndrome is a form of severe epilepsy that begins in childhood. It is characterized by multiple types of seizures and intellectual disability. People with Lennox-Gastaut syndrome begin having frequent seizures in early chi...
4. `medquad_ans_0001024-1__dup1` (score=0.0161) - A CDKL5-related disorder is a genetic, neuro-developmental condition due to changes (mutations) in the CDKL5 gene. Epileptic encephalopathy (epilepsy accompanied by cognitive and behavioral problems) is the core symptom of a CDKL5-relate...
5. `medquad_ans_0000592-4` (score=0.0159) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of some type of epilepsy. ...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. In about one-third of cases, the cause is unknown and described as cryptogenic, meaning there is no identifiable underlying reason [medquad_ans_0000592-3]. While Lennox-Gastaut syndrome can occur as part of a genetic disorder such as tuberous sclerosis complex, no specific gene mutations have been identified that are uniquely associated with the syndrome [medquad_ans_0000592-3]. Therefore, there are no known genetic changes specifically linked to Lennox-Gastaut syndrome.

**Citations detected**: `medquad_ans_0000592-3`
**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

## Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.0318) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-3` (score=0.0164) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step production (synthesi...
3. `medquad_ans_0000359-2` (score=0.0164) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
4. `medquad_ans_0000066-1` (score=0.0161) - Arginine:glycine amidinotransferase deficiency is an inherited disorder that primarily affects the brain. People with this disorder have mild to moderate intellectual disability and delayed speech development. Some affected individuals d...
5. `medquad_ans_0000066-5` (score=0.0159) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotransferase deficiency T...

**Answer**:

Arginine:glycine amidinotransferase deficiency affects about 1 per million people worldwide [medquad_ans_0000359-2]. It is more common in certain areas of the province of Quebec, Canada [medquad_ans_0000359-2].

**Citations detected**: `medquad_ans_0000359-2`
**Supported citations**: `medquad_ans_0000359-2`
**Unsupported citations**: _none_

## Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Top docs**:

1. `medquad_ans_0002401-1` (score=0.0311) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the table below includes...
2. `medquad_ans_0002399-2` (score=0.0294) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, although almost all affect...
3. `medquad_ans_0000169-1` (score=0.0271) - What are the signs and symptoms of ADULT syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for ADULT syndrome. If the information is available, the table below includes how often the symptom is seen...
4. `medquad_ans_0001971-1` (score=0.0223) - What are the signs and symptoms of Dyskeratosis congenita autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal dominant. If the information is available, ...
5. `medquad_ans_0003781-1` (score=0.0211) - What are the signs and symptoms of Mandibuloacral dysplasia with type A lipodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Mandibuloacral dysplasia with type A lipodystrophy. If the informat...

**Answer**:

The symptoms of Focal dermal hypoplasia include streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These skin features can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths, called papillomas, around the nostrils, lips, anus, and female genitalia. They may also be present in the throat, specifically in the esophagus or larynx, and can cause problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails as well as sparse, brittle, or absent scalp hair. [medquad_ans_0002399-2]

**Citations detected**: `medquad_ans_0002399-2`
**Supported citations**: `medquad_ans_0002399-2`
**Unsupported citations**: _none_

## Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.0328) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
2. `medquad_ans_0000418-5__dup1` (score=0.0290) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild game meat. Laborato...
3. `medquad_ans_0004252-2` (score=0.0161) - What causes myelodysplastic/myeloproliferative disease? In most cases, the cause of myelodysplastic/myeloproliferative disease is unknown, and there is limited information regarding potential causes. No specific genetic defects have been...
4. `medquad_ans_0000418-1__dup2` (score=0.0161) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (meat-eating) animals...
5. `medquad_ans_0000014_3-3` (score=0.0159) - Key Points - Avoiding risk factors and increasing protective factors may help prevent cancer. - The following risk factors increase the risk of endometrial cancer: - Endometrial hyperplasia - Estrogen - Tamoxifen - Obesity, weight gain, ...

**Answer**:

People at risk for trichinellosis include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking puts individuals at risk. Additionally, individuals who consume undercooked wild game, especially bear meat, are at risk. The risk from commercially raised and properly prepared pork is very low, but it remains a concern with raw or undercooked wild game meats [medquad_ans_0000418-2__dup1].

**Citations detected**: `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.0318) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how often the symptom is se...
2. `medquad_ans_0006607-2` (score=0.0299) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, the table below includ...
3. `medquad_ans_0000108-3__dup1` (score=0.0164) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of cognitive impairment a...
4. `medquad_ans_0000108-1__dup2` (score=0.0161) - Dravet syndrome, also called severe myoclonic epilepsy of infancy (SMEI), is a severe form of epilepsy. It appears during the first year of life with frequent febrile seizures fever-related seizures that, by definition, are rare beyond a...
5. `medquad_ans_0004898-1` (score=0.0161) - What are the signs and symptoms of Pitt-Hopkins-like syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pitt-Hopkins-like syndrome. If the information is available, the table below includes how o...

**Answer**:

The signs and symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Citations detected**: `medquad_ans_0001924-2`
**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

## Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Top docs**:

1. `medquad_ans_0000826-4` (score=0.0318) - Familial forms of prion disease are inherited in an autosomal dominant pattern, which means one copy of the altered PRNP gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the altered gene f...
2. `medquad_ans_0000826-3` (score=0.0313) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, which have overlapping ...
3. `medquad_ans_0000011-13` (score=0.0306) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten CJD from medical proc...
4. `medquad_ans_0002316-1` (score=0.0306) - Fatal familial insomnia (FFI) is an inherited prion disease that affects the brain and other parts of the nervous system. Prion diseases, also known as transmissible spongiform encephalopathies (TSE), are a group of rare neurodegenerativ...
5. `medquad_ans_0002533-3` (score=0.0301) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although the exact function ...

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which account for between 10 and 15 percent of all cases of prion disease. These mutations result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. This abnormal protein can bind to the normal prion protein (PrPC) and promote its transformation into PrPSc, leading to the accumulation of PrPSc in the brain and neuronal damage [medquad_ans_0000826-3]. Familial forms of prion disease, such as familial Creutzfeldt-Jakob disease (CJD), Gerstmann-Strussler-Scheinker syndrome (GSS), and fatal familial insomnia (FFI), are caused by these PRNP gene mutations and are inherited in an autosomal dominant pattern [medquad_ans_0000826-4]. In some

**Citations detected**: `medquad_ans_0000826-3`, `medquad_ans_0000826-4`
**Supported citations**: `medquad_ans_0000826-3`, `medquad_ans_0000826-4`
**Unsupported citations**: _none_

## Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.0328) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information is available, the ta...
2. `medquad_ans_0006429-1` (score=0.0310) - What are the signs and symptoms of Weyers acrofacial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers acrofacial dysostosis. If the information is available, the table below includes h...
3. `medquad_ans_0006470-1` (score=0.0290) - What are the signs and symptoms of WT limb blood syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for WT limb blood syndrome. If the information is available, the table below includes how often the...
4. `medquad_ans_0002743-1` (score=0.0278) - What are the signs and symptoms of Heart-hand syndrome, Spanish type? The Human Phenotype Ontology provides the following list of signs and symptoms for Heart-hand syndrome, Spanish type. If the information is available, the table below ...
5. `medquad_ans_0000902-1__dup1` (score=0.0252) - What are the signs and symptoms of Camptobrachydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptobrachydactyly. If the information is available, the table below includes how often the sympt...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Citations detected**: `medquad_ans_0006430-1`
**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

## Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.0328) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the table below includes h...
2. `medquad_ans_0004524-1` (score=0.0318) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how often the symptom is ...
3. `medquad_ans_0004961-1` (score=0.0214) - What are the signs and symptoms of Polyneuropathy mental retardation acromicria premature menopause? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyneuropathy mental retardation acromicria prematur...
4. `medquad_ans_0002060-1` (score=0.0161) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the information is availab...
5. `medquad_ans_0006422-1` (score=0.0159) - Werner's syndrome is a disease chiefly characterized by premature aging and cancer predisposition. Development is typically normal until the end of the first decade; the first sign is the lack of a growth spurt during puberty. Early sign...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which often draws from Orphanet, and represents rough estimates of the percentage of patients who have each feature [medquad_ans_0005029-1].

**Citations detected**: `medquad_ans_0005029-1`
**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

## Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Top docs**:

1. `medquad_ans_0004919-1` (score=0.0164) - Platelet storage pool deficiency refers to a group of conditions that are caused by problems with the platelet granules. Platelet granules are tiny storage sacs found within the platelets which release various substances to help stop ble...
2. `medquad_ans_0000934-2` (score=0.0164) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
3. `medquad_ans_0000979-1` (score=0.0161) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (platelets). This plate...
4. `medquad_ans_0000989-2` (score=0.0161) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0001079-3` (score=0.0159) - Most people with a 1q21.1 microdeletion are missing a sequence of about 1.35 million DNA building blocks (base pairs), also written as 1.35 megabases (Mb), in the q21.1 region of chromosome 1. However, the exact size of the deleted regio...

**Answer**:

Thrombocytopenia-absent radius (TAR) syndrome affects an estimated 1 in 50,000 people [medquad_ans_0000989-2].

**Citations detected**: `medquad_ans_0000989-2`
**Supported citations**: `medquad_ans_0000989-2`
**Unsupported citations**: _none_

## Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Top docs**:

1. `medquad_ans_0000019_3-7` (score=0.0284) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, taking part in a clini...
2. `medquad_ans_0000007_2-6` (score=0.0164) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
3. `medquad_ans_0000005-6__dup1` (score=0.0164) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports research studies on ...
4. `medquad_ans_0000009_2-8` (score=0.0161) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ongoing clinical tri...
5. `medquad_ans_0000214-4__dup1` (score=0.0161) - The NINDS conducts and supports a wide range of studies that explore the complex systems of brain development. These studies include the identification of the mechanism of action of the known causes of NMD as well as studies to identify ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Tay-Sachs Disease. [medquad_ans_0000019_3-7] and [medquad_ans_0000007_2-6] discuss general clinical trials for cancer treatments and liver transplants, but they do not reference Tay-Sachs Disease or specific research related to it. Therefore, based on the given information, it is not possible to answer the question.

**Citations detected**: `medquad_ans_0000019_3-7`, `medquad_ans_0000007_2-6`
**Supported citations**: `medquad_ans_0000019_3-7`, `medquad_ans_0000007_2-6`
**Unsupported citations**: _none_

## Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.0328) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder...
2. `medquad_ans_0000050-1__dup1` (score=0.0318) - What are the signs and symptoms of Achromatopsia 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Achromatopsia 3. If the information is available, the table below includes how often the symptom is se...
3. `medquad_ans_0000008-3` (score=0.0317) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the retina, which is the ...
4. `medquad_ans_0000008-2` (score=0.0315) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on one of the Eastern Ca...
5. `medquad_ans_0000008-5` (score=0.0305) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources from MedlinePlus offe...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green. The condition results from changes in one of several

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Top docs**:

1. `medquad_ans_0000544-1__dup1` (score=0.0323) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments are not helping you a...
2. `medquad_ans_0000049-4__dup3` (score=0.0313) - Treatment Goals: Manage Pain, Improve Function Osteoarthritis treatment plans often include ways to manage pain and improve function. Such plans can include exercise, rest and joint care, pain relief, weight control, medicines, surgery, ...
3. `medquad_ans_0000039-11` (score=0.0313) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bear - walking aids suc...
4. `medquad_ans_0000039-9` (score=0.0301) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
5. `medquad_ans_0000039-26` (score=0.0278) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circle Bethesda, MD 20892...

**Answer**:

The provided context does not include information about treatments for knee replacement. It describes knee replacement surgery itself, including its purpose, procedure, and potential risks, but does not list or describe treatments following or related to knee replacement. Therefore, based on the given context, the treatments for knee replacement cannot be determined.

[medquad_ans_0000544-1__dup1]

**Citations detected**: `medquad_ans_0000544-1__dup1`
**Supported citations**: `medquad_ans_0000544-1__dup1`
**Unsupported citations**: _none_

## Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Top docs**:

1. `medquad_ans_0004775-2` (score=0.0227) - What are the signs and symptoms of Pearson syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pearson syndrome. If the information is available, the table below includes how often the symptom is ...
2. `medquad_ans_0002259-1` (score=0.0164) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is available, the tabl...
3. `medquad_ans_0001692-2` (score=0.0164) - What are the signs and symptoms of Cushing's syndrome? The signs and symptoms of Cushing's syndrome may include: Upper body obesity Severe fatigue Muscle weakness High blood pressure Backache Elevated blood sugar Easy bruising Bluish-red...
4. `medquad_ans_0000090-4__dup2` (score=0.0161) - People at greatest risk for metabolic syndrome have these underlying causes: Abdominal obesity (a large waistline) An inactive lifestyle Insulin resistance Some people are at risk for metabolic syndrome because they take medicines that c...
5. `medquad_ans_0005690-2` (score=0.0161) - What are the signs and symptoms of Sotos syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sotos syndrome. If the information is available, the table below includes how often the symptom is seen...

**Answer**:

The provided context does not contain information about the symptoms of Metabolic Syndrome. [medquad_ans_0004775-2] and [medquad_ans_0002259-1] discuss symptoms of Pearson syndrome and Familial hyperaldosteronism type III, respectively, but neither addresses Metabolic Syndrome. Therefore, no answer can be provided based on the given context.

**Citations detected**: `medquad_ans_0004775-2`, `medquad_ans_0002259-1`
**Supported citations**: `medquad_ans_0004775-2`, `medquad_ans_0002259-1`
**Unsupported citations**: _none_

## Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Top docs**:

1. `medquad_ans_0000108-2__dup1` (score=0.0164) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
2. `medquad_ans_0005211-1` (score=0.0164) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0000549-1__dup1` (score=0.0161) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is available, the tabl...
4. `medquad_ans_0005210-2` (score=0.0161) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how often the symptom is se...
5. `medquad_ans_0002259-1` (score=0.0159) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is available, the tabl...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, and thickened nuchal skin fold. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_005211-1]. Additionally, the condition is inherited in an autosomal recessive manner, and stillbirth has been observed in some cases [medquad_ans_005211-1].

**Citations detected**: `medquad_ans_005211-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_005211-1`

## Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Top docs**:

1. `medquad_ans_0003966-1` (score=0.0320) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to saposin B deficiency. If ...
2. `medquad_ans_0003965-2` (score=0.0300) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the table below includes h...
3. `medquad_ans_0001799-1` (score=0.0269) - What are the signs and symptoms of Dermatoleukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatoleukodystrophy. If the information is available, the table below includes how often the s...
4. `medquad_ans_0003070-2` (score=0.0257) - What are the signs and symptoms of Hypomyelination with atrophy of basal ganglia and cerebellum? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypomyelination with atrophy of basal ganglia and cerebel...
5. `medquad_ans_0003573-2` (score=0.0253) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy with brain ...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, and limitation of joint mobility [medquad_ans_0003965-2].

**Citations detected**: `medquad_ans_0003965-2`
**Supported citations**: `medquad_ans_0003965-2`
**Unsupported citations**: _none_

## Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Top docs**:

1. `medquad_ans_0000186-1` (score=0.0164) - Chronic granulomatous disease is a disorder that causes the immune system to malfunction, resulting in a form of immunodeficiency. Immunodeficiencies are conditions in which the immune system is not able to protect the body from foreign ...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000186-4` (score=0.0161) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who...
4. `medquad_ans_0001014-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000186-5` (score=0.0159) - These resources address the diagnosis or management of chronic granulomatous disease: - American Academy of Allergy, Asthma, and Immunology - Gene Review: Gene Review: Chronic Granulomatous Disease - Genetic Testing Registry: Chronic gra...

**Answer**:

The provided context does not state whether chronic granulomatous disease is inherited. Therefore, based on the given information, it cannot be determined if the condition is inherited. [medquad_ans_0000186-1]

**Citations detected**: `medquad_ans_0000186-1`
**Supported citations**: `medquad_ans_0000186-1`
**Unsupported citations**: _none_

## Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000061-3__dup1` (score=0.0318) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likelihood that neurological...
2. `medquad_ans_0000061-2__dup1` (score=0.0299) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic resonance imaging (M...
3. `medquad_ans_0000061-1__dup1` (score=0.0276) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to and from parts of the...
4. `medquad_ans_0000061-4__dup1` (score=0.0269) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research focuses on finding ...
5. `medquad_ans_0000222-1__dup1` (score=0.0190) - Paresthesia refers to a burning or prickling sensation that is usually felt in the hands, arms, legs, or feet, but can also occur in other parts of the body. The sensation, which happens without warning, is usually painless and described...

**Answer**:

The outlook for central cord syndrome varies, but most people who have the condition due to trauma experience some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. Those who receive medical intervention soon after injury often have good outcomes. Many individuals recover substantial function after their initial injury, with the ability to walk being recovered in most cases, although some impairment may remain. Improvement typically occurs first in the legs, then the bladder, and may be seen in the arms, with hand function recovering last, if at all. Recovery is generally better in younger patients compared to those over the age of 50 [medquad_ans_0000061-3__dup1]. Additionally, early surgery may improve recovery chances, and surgery can be beneficial in cases of persistent spinal cord compression and ongoing neurological deterioration [medquad_ans_0000061-2__dup1]. There is

**Citations detected**: `medquad_ans_0000061-3__dup1`, `medquad_ans_0000061-2__dup1`
**Supported citations**: `medquad_ans_0000061-3__dup1`, `medquad_ans_0000061-2__dup1`
**Unsupported citations**: _none_

## Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Top docs**:

1. `medquad_ans_0000261-3` (score=0.0320) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnormalities, and gastro...
2. `medquad_ans_0000094-13` (score=0.0318) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. - Th...
3. `medquad_ans_0000094-1__dup6` (score=0.0315) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. The ep...
4. `medquad_ans_0000094-4__dup3` (score=0.0315) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting syndrome, which occu...
5. `medquad_ans_0000261-1` (score=0.0311) - Cyclic vomiting syndrome is a disorder that causes recurrent episodes of nausea, vomiting, and tiredness (lethargy). This condition is diagnosed most often in young children, but it can affect people of any age. The episodes of nausea, v...

**Answer**:

The causes of cyclic vomiting syndrome have not been fully determined, but researchers have proposed several contributing factors. These include changes in brain function, hormonal abnormalities, gastrointestinal problems, and abnormalities of the autonomic nervous system. Some researchers believe it is a migraine-like condition related to changes in signaling between nerve cells in certain areas of the brain. Additionally, in some cases—particularly those beginning in childhood—changes in mitochondrial DNA have been associated with the disorder. These mitochondrial DNA changes may impair energy production in cells, potentially leading to dysfunction in the autonomic nervous system and affecting the digestive system. However, it remains unclear how these mitochondrial changes specifically lead to symptoms such as nausea, vomiting, lethargy, abdominal pain, or migraines [medquad_ans_0000261-3]. The exact cause is still unknown [medquad_ans_0000094-13].

**Citations detected**: `medquad_ans_0000261-3`, `medquad_ans_0000094-13`
**Supported citations**: `medquad_ans_0000261-3`, `medquad_ans_0000094-13`
**Unsupported citations**: _none_

## Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Top docs**:

1. `medquad_ans_0000875-1__dup1` (score=0.0320) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. But feeling very sad,...
2. `medquad_ans_0000071-1__dup3` (score=0.0294) - Is it hard for your child to sit still? Does your child act without thinking first? Does your child start but not finish things? If so, your child may have attention deficit hyperactivity disorder (ADHD). Nearly everyone shows some of th...
3. `medquad_ans_0000180-1__dup2` (score=0.0293) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not always easy to know when...
4. `medquad_ans_0000432-1__dup2` (score=0.0256) - Summary : Health literacy refers to how well a person can get the health information and services that they need, and how well they understand them. It is also about using them to make good health decisions. It involves differences that ...
5. `medquad_ans_0000093-1__dup3` (score=0.0240) - Bipolar disorder is a serious mental illness. People who have it go through unusual mood changes. They go from very happy, "up," and active to very sad and hopeless, "down," and inactive, and then back again. They often have normal moods...

**Answer**:

Yes, there is information about child mental health. The context indicates that mental health problems in children are real, painful, and sometimes severe, and early recognition and treatment are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat [medquad_ans_0000180-1__dup2]. Warning signs of a more serious mental health problem include problems in multiple settings, changes in appetite or sleep, social withdrawal, fear of previously non-fearful things, return to younger behaviors like bedwetting, signs of sadness or tearfulness, and self-destructive behavior [medquad_ans_0000180-1__dup2]. Additionally, symptoms such as feeling very sad, hopeless, or worthless, frequent anger or worry, grief lasting a long time after a loss, feeling that the mind is out of control, substance use, obsessive exercise or eating,

**Citations detected**: `medquad_ans_0000180-1__dup2`
**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

## Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.0311) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected by Cowden syndrome wi...
2. `medquad_ans_0000246-5` (score=0.0164) - These resources address the diagnosis or management of Cowden syndrome: - Gene Review: Gene Review: PTEN Hamartoma Tumor Syndrome (PHTS) - Genetic Testing Registry: Cowden syndrome - Genetic Testing Registry: Cowden syndrome 1 - Genetic ...
3. `medquad_ans_0000246-1` (score=0.0161) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas. These growths are m...
4. `medquad_ans_0000740-2__dup1` (score=0.0161) - What are the signs and symptoms of Birt-Hogg-Dube syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Birt-Hogg-Dube syndrome. If the information is available, the table below includes how often t...
5. `medquad_ans_0001618-3` (score=0.0159) - What causes Cowden syndrome? Most cases of Cowden syndrome are caused by changes (mutations) in the PTEN gene. PTEN is a tumor suppressor gene which means that it encodes a protein that helps keep cells from growing and dividing too rapi...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority also develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; an enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.0325) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscular Dystrophy - Geneti...
2. `medquad_ans_0000576-5` (score=0.0313) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystrophy - Genetic Testi...
3. `medquad_ans_0006267-3` (score=0.0236) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often becomes necessa...
4. `medquad_ans_0001026-5` (score=0.0226) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrophy These resources f...
5. `medquad_ans_0000205-4__dup1` (score=0.0214) - The NINDS supports a broad program of research studies on MD. The goals of these studies are to understand MD and to develop techniques to diagnose, treat, prevent, and ultimately cure the disorder. The NINDS is a member of the Muscular ...

**Answer**:

The provided context does not specify the treatments for Fukuyama congenital muscular dystrophy. While it lists resources related to the diagnosis and management of Fukuyama congenital muscular dystrophy, including gene reviews, genetic testing registry entries, and MedlinePlus encyclopedia entries, no specific treatment options are detailed in the context. [medquad_ans_0000395-5]

**Citations detected**: `medquad_ans_0000395-5`
**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

## Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Top docs**:

1. `medquad_ans_0000004_6-6` (score=0.0313) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spreads in the body. - Sta...
2. `medquad_ans_0000004_7-1` (score=0.0293) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major types of childhood n...
3. `medquad_ans_0000004_6-8` (score=0.0283) - Key Points - There are different types of treatment for children with Hodgkin lymphoma. - Children with Hodgkin lymphoma should have their treatment planned by a team of health care providers who are experts in treating childhood cancer....
4. `medquad_ans_0000004_6-4` (score=0.0282) - Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of health, in...
5. `medquad_ans_0000004_7-6` (score=0.0279) - Key Points - After childhood non-Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spreads in the body. -...

**Answer**:

Childhood Hodgkin lymphoma is diagnosed through a series of tests that determine if cancer cells have spread within the lymph system or to other parts of the body. This process is called staging and includes the following tests and procedures: CT scan (CAT scan), which creates detailed pictures of areas inside the body using x-rays and possibly a dye; PET scan (positron emission tomography scan), which detects malignant tumor cells by measuring glucose uptake, as cancer cells take up more glucose than normal cells; MRI (magnetic resonance imaging), which uses magnets, radio waves, and a computer to produce detailed images of body areas; and bone marrow aspiration and biopsy, where bone marrow and a small piece of bone are removed and examined under a microscope for abnormal cells [medquad_ans_0000004_6-6].

**Citations detected**: `medquad_ans_0000004_6-6`
**Supported citations**: `medquad_ans_0000004_6-6`
**Unsupported citations**: _none_

## Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Top docs**:

1. `medquad_ans_0003361-1` (score=0.0164) - KBG syndrome is a rare condition characterized mainly by skeletal abnormalities, distinctive facial features, and intellectual disability. Specific signs and symptoms may include delayed bone age; abnormalities of the bones of the spine,...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000556-1` (score=0.0161) - KBG syndrome is a rare disorder that affects several body systems. "KBG" represents the surname initials of the first families diagnosed with the disorder. Common signs and symptoms in individuals with this condition include unusual faci...
4. `medquad_ans_0000572-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000556-3` (score=0.0159) - KBG syndrome is caused by mutations in the ANKRD11 gene. The protein produced from this gene enables other proteins to interact with each other and helps control gene activity. The ANKRD11 protein is found in nerve cells (neurons) in the...

**Answer**:

Yes, KBG syndrome can be inherited, but the pattern of inheritance is not uniform. In some cases, it is caused by a mutation in the ANKRD11 gene and is inherited in an autosomal dominant manner, meaning only one copy of the mutated gene is sufficient to cause the condition [medquad_ans_0003361-1]. In other cases, the genetic cause is unclear, and the condition may occur sporadically [medquad_ans_0003361-1]. However, one source incorrectly states that the condition is inherited in an autosomal recessive pattern, which would require two copies of the mutated gene [medquad_ans_0000785-4]; this appears to be a misattribution or error, as KBG syndrome is more consistently described as autosomal dominant or sporadic [medquad_ans_0003361-1].

**Citations detected**: `medquad_ans_0003361-1`, `medquad_ans_0000785-4`
**Supported citations**: `medquad_ans_0003361-1`, `medquad_ans_0000785-4`
**Unsupported citations**: _none_

## Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Top docs**:

1. `medquad_ans_0000029-1` (score=0.0325) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. However, about 20 percent ...
2. `medquad_ans_0000197-1__dup1` (score=0.0310) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
3. `medquad_ans_0000196-1__dup1` (score=0.0303) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
4. `medquad_ans_0000029-4` (score=0.0303) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an au...
5. `medquad_ans_0000194-1__dup1` (score=0.0303) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...

**Answer**:

Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns do not show signs or symptoms at birth, but about 20 percent are born with features such as enlarged liver and spleen (hepatosplenomegaly), elevated liver enzymes, decreased blood platelets (thrombocytopenia), and abnormal neurological responses. This combination is typically associated with an immune response to congenital viral infection, although no actual infection is found, leading to the syndrome being referred to as a "mimic of congenital infection." Within the first year of life, most individuals experience severe brain dysfunction (encephalopathy), characterized by irritability, poor feeding, intermittent fevers (sterile pyrexias), seizures, developmental regression, loss of acquired skills, slowed brain and skull growth (microcephaly), and inflammation detectable in cerebrospinal fluid.

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Top docs**:

1. `medquad_ans_0000026_2-8` (score=0.0323) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
2. `medquad_ans_0000026_1-7` (score=0.0323) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy - New types of treat...
3. `medquad_ans_0000006_3-8` (score=0.0261) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000026_2-7` (score=0.0255) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tested in clinical tria...
5. `medquad_ans_0000040_1-6` (score=0.0252) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

Research being done for Renal Cell Cancer includes clinical trials studying chemotherapy or biologic therapy using drugs placed directly into the renal pelvis or the ureter, which is referred to as regional chemotherapy and regional biologic therapy [medquad_ans_0000026_2-8]. These treatments aim to target cancer cells in specific areas of the body, such as the renal pelvis or ureter, using drugs or biologic substances that boost or direct the body's immune response against cancer. Clinical trials are conducted to determine if these new treatments are safe, effective, or better than standard treatments. Patients may participate in these trials before, during, or after starting their cancer treatment, and their involvement helps advance future cancer treatment options [medquad_ans_0000026_2-8].

**Citations detected**: `medquad_ans_0000026_2-8`
**Supported citations**: `medquad_ans_0000026_2-8`
**Unsupported citations**: _none_

## Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Top docs**:

1. `medquad_ans_0000190-6` (score=0.0318) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the progression of live...
2. `medquad_ans_0000190-5__dup1` (score=0.0315) - A health care provider may use the following tests to diagnose primary biliary cirrhosis: - a medical and family history - a physical exam - blood tests - imaging tests - a liver biopsy A health care provider usually bases a diagnosis of...
3. `medquad_ans_0000190-9` (score=0.0313) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxic wastes build up in...
4. `medquad_ans_0000190-3__dup2` (score=0.0295) - The first and most common symptoms of primary biliary cirrhosis are - fatigue, or feeling tired - itching skin, and darkened skin in itching areas due to scratching - dry eyes and mouth Some people may have jaundice, a condition that cau...
5. `medquad_ans_0000190-1__dup3` (score=0.0286) - Primary biliary cirrhosis is a chronic, or long lasting, disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. The bile ducts carry a fluid called bile from the liver to the gallbl...

**Answer**:

Treatment for primary biliary cirrhosis depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. The primary medication prescribed is ursodiol (Actigall, Urso), a nontoxic bile acid that replaces more toxic bile acids produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival, although it does not cure the disease. Treatment benefits are greatest when started early, but ursodiol can still slow disease progression even when used later.

Patients should avoid alcohol and illegal substances, as these can worsen liver damage. They should also avoid complementary and alternative medications, such as herbs, and should consult a healthcare provider before taking any new prescription, over-the-counter medications, or vitamins, as many can affect liver

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.0328) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People who have seizure dis...
2. `medquad_ans_0000916-1__dup2` (score=0.0315) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of Earth. A tsunami ca...
3. `medquad_ans_0003663-2` (score=0.0303) - What are the signs and symptoms of Long QT syndrome? Signs and symptoms of the arrhythmias experienced by people with long QT syndrome includes unexplained fainting, seizures, drowning or near drowning, and sudden cardiac arrest or death...
4. `medquad_ans_0001853-3` (score=0.0161) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...
5. `medquad_ans_0000023-1__dup1` (score=0.0161) - Cerebral hypoxia refers to a condition in which there is a decrease of oxygen supply to the brain even though there is adequate blood flow. Drowning, strangling, choking, suffocation, cardiac arrest, head trauma, carbon monoxide poisonin...

**Answer**:

Drowning is a condition that occurs when a person takes in too much water, particularly into the lungs, which can lead to respiratory failure and death. It can happen quickly and silently, even with as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at increased risk in the water. Drowning is also noted as the most common cause of death related to a tsunami [medquad_ans_0000299-1__dup2]. Additionally, unexplained fainting, seizures, drowning, or near drowning are signs and symptoms associated with long QT syndrome [medquad_ans_0003663-2].

**Citations detected**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0003663-2`
**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0003663-2`
**Unsupported citations**: _none_

## Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Top docs**:

1. `medquad_ans_0002058-3` (score=0.0320) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules which allow collagen ...
2. `medquad_ans_0002058-6` (score=0.0318) - How might Ehlers-Danlos syndrome, kyphoscoliosis type be treated? The treatment of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is focused on preventing serious complications and relieving associated signs and symptoms. For example,...
3. `medquad_ans_0002058-5` (score=0.0315) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tests may then be recom...
4. `medquad_ans_0002058-1` (score=0.0308) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragile and bruises easil...
5. `medquad_ans_0002058-2` (score=0.0299) - What are the signs and symptoms of Ehlers-Danlos syndrome, kyphoscoliosis type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type vary but may include: Hyperextensible skin that is fragile and bruises easily Joi...

**Answer**:

Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen, specifically due to mutations in the PLOD1 gene. This gene encodes an enzyme that helps process molecules allowing collagen to form stable interactions. Mutations in the PLOD1 gene result in reduced levels of functional enzyme, disrupting collagen networks and weakening connective tissues. This leads to characteristic signs and symptoms such as hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, and progressive kyphoscoliosis [medquad_ans_0002058-3, medquad_ans_0002058-1]. Diagnosis is typically based on clinical signs and symptoms, and may be confirmed through urine tests, skin biopsy, or genetic testing for mutations in the PLOD1 gene [medquad

**Citations detected**: `medquad_ans_0002058-3, medquad_ans_0002058-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0002058-3, medquad_ans_0002058-1`

## Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.0328) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is available, the tabl...
2. `medquad_ans_0006567-2` (score=0.0306) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is available, the table ...
3. `medquad_ans_0004302-1` (score=0.0296) - What are the signs and symptoms of NADH cytochrome B5 reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for NADH cytochrome B5 reductase deficiency. If the information is available, the ...
4. `medquad_ans_0001612-1` (score=0.0254) - What are the signs and symptoms of Cortisone reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Cortisone reductase deficiency. If the information is available, the table below includ...
5. `medquad_ans_0006569-2` (score=0.0225) - What are the signs and symptoms of 21-hydroxylase deficiency? Symptoms can vary greatly from patient to patient with 21-hydroxylase deficiency, as a result distinct forms of this deficiency have been recognized. Three common forms includ...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Citations detected**: `medquad_ans_0001860-2`
**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

## Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.0323) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000669-1` (score=0.0311) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin word for necklace (m...
3. `medquad_ans_0000669-4` (score=0.0227) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suf...
4. `medquad_ans_0000669-3` (score=0.0161) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for making proteins that gi...
5. `medquad_ans_0000359-2` (score=0.0161) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.

**Answer**:

The prevalence of monilethrix is unknown. [medquad_ans_0000669-2]

**Citations detected**: `medquad_ans_0000669-2`
**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

## Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.0328) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the results of newborn s...
2. `medquad_ans_0002733-2` (score=0.0323) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headaches,and hypotonia Psy...
3. `medquad_ans_0001853-3` (score=0.0159) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...
4. `medquad_ans_0002890-1` (score=0.0159) - Hirschsprung disease is a disease of the large intestine or colon. People with this disease do not have the nerve cells in the intestine required to expel stools from the body normally. Symptoms of Hirschsprung disease usually show up in...
5. `medquad_ans_0000108-2__dup1` (score=0.0156) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. The condition may be diagnosed through newborn screening tests. Most individuals are asymptomatic, but those who do show symptoms typically experience onset between ages 3 and 9, although it can occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Top docs**:

1. `medquad_ans_0000683-3` (score=0.0259) - Mutations in the GUSB gene cause MPS VII. This gene provides instructions for producing the beta-glucuronidase (-glucuronidase) enzyme, which is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). GAGs we...
2. `medquad_ans_0000428-5` (score=0.0164) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resources from MedlinePlus ...
3. `medquad_ans_0002615-4` (score=0.0161) - How is glycogen storage disease type 13 diagnosed? Glycogen storage disease type 13 is diagnosed by taking a sample of muscle tissue (muscle biopsy) to determine if there is enough of the chemical enolase working in the muscle cells. Gen...
4. `medquad_ans_0000342-4` (score=0.0161) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0002615-3` (score=0.0159) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an important source of energ...

**Answer**:

The provided context does not contain information about the genetic changes related to glycogen storage disease type VII. The context discusses glycogen storage disease type 13, which is caused by mutations in the ENO3 gene, and mentions mutations in the GUSB gene in relation to MPS VII (Mucopolysaccharidosis type VII), which is a different condition. Therefore, based on the given information, the genetic changes for glycogen storage disease type VII cannot be determined.

Answer: Insufficient context to determine the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5]

**Citations detected**: `medquad_ans_0000428-5`
**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

## Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.0325) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megaloblastic anemia, thi...
2. `medquad_ans_0006061-1` (score=0.0290) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this condition between i...
3. `medquad_ans_0000977-2` (score=0.0278) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
4. `medquad_ans_0000977-1` (score=0.0261) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low number of red blood cell...
5. `medquad_ans_0000172-2__dup1` (score=0.0161) - Treatment is symptomatic and supportive, and may include drug therapy.

**Answer**:

The treatment for thiamine-responsive megaloblastic anemia syndrome is high doses of vitamin B1 (thiamine), as the anemia can be treated with thiamine [medquad_ans_0006061-1]. However, it remains unclear whether thiamine treatment can improve hearing loss or prevent hearing loss [medquad_ans_0000977-1]. Individuals with this syndrome usually require insulin to treat their diabetes [medquad_ans_0000977-1].

**Citations detected**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`
**Supported citations**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`
**Unsupported citations**: _none_

## Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000587-4__dup1` (score=0.0308) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancements have been made in...
2. `medquad_ans_0004939-2` (score=0.0308) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their ability to filter waste pr...
3. `medquad_ans_0004939-1` (score=0.0304) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the their ability to fil...
4. `medquad_ans_0000804-1` (score=0.0301) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the blood. The growth o...
5. `medquad_ans_0000804-5` (score=0.0286) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Recessive - Gene...

**Answer**:

Medical management of Polycystic Kidney Disease is currently symptomatic and involves supportive care. This includes mechanical ventilation to treat lung underdevelopment and breathing issues due to enlarged kidneys, dialysis when infants produce little or no urine (oliguria or anuria), treatment of low sodium levels (hyponatremia) with diuresis and/or sodium supplementation, and management of high blood pressure (hypertension) with medication. Kidney failure may require dialysis or kidney transplantation. Growth failure may be managed with gastrostomy tubes or growth hormone therapy. Urinary tract infections are treated with antibiotics. In cases of liver involvement, a shunt may be required to manage progressive high blood pressure, and liver transplantation may be necessary. [medquad_ans_0000587-4__dup1]

**Citations detected**: `medquad_ans_0000587-4__dup1`
**Supported citations**: `medquad_ans_0000587-4__dup1`
**Unsupported citations**: _none_

## Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.0320) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research efforts are focused ...
2. `medquad_ans_0000025_1-8` (score=0.0262) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
3. `medquad_ans_0000032_1-8` (score=0.0253) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
4. `medquad_ans_0000006_3-8` (score=0.0249) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...
5. `medquad_ans_0000003_5-7` (score=0.0243) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly the role of infectious agents such as viruses and bacteria in triggering the disease in individuals with a genetic predisposition. Scientists are investigating these infectious agents to understand their mechanisms and explore potential new therapies [medquad_ans_0000061-5__dup2]. Additionally, researchers are studying the complex interactions between the hormonal, nervous, and immune systems to explain why more women than men develop rheumatoid arthritis, including how changes in steroid hormones like estrogen and testosterone may influence disease development or flares. Another area of interest is the improvement of rheumatoid arthritis during pregnancy, which may provide clues about hormonal influences [medquad_ans_0000061-5__dup2].

Clinical trials for rheumatoid arthritis are also being conducted, with information available through Clinical Trials.gov, where users can search for current trials by typing "r

**Citations detected**: `medquad_ans_0000061-5__dup2`
**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

## Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.0325) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the ...
2. `medquad_ans_0002561-1` (score=0.0285) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an abnormally-shaped hea...
3. `medquad_ans_0002561-2` (score=0.0271) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms seen in the five affec...
4. `medquad_ans_0002561-3` (score=0.0259) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
5. `medquad_ans_0000071-2__dup1` (score=0.0161) - There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is surgery to correct the craniosynostosis. Surgery is typically performed when the person is between 9 to 12 months of age and aims to increase the space in the anterior part of the skull. If other sutures besides the coronal suture are involved, additional surgeries may be required [medquad_ans_0002561-4]. Treatment has included surgical intervention to address the craniosynostosis, and no issues with development or normal intelligence have been reported [medquad_ans_0002561-1].

**Citations detected**: `medquad_ans_0002561-4`, `medquad_ans_0002561-1`
**Supported citations**: `medquad_ans_0002561-4`, `medquad_ans_0002561-1`
**Unsupported citations**: _none_

## Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Top docs**:

1. `medquad_ans_0000868-5` (score=0.0325) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: Center for Genetic D...
2. `medquad_ans_0000868-1` (score=0.0164) - Renpenning syndrome is a disorder that almost exclusively affects males, causing developmental delay, moderate to severe intellectual disability, and distinctive physical features. Individuals with Renpenning syndrome typically have shor...
3. `medquad_ans_0000184-5` (score=0.0161) - These resources address the diagnosis or management of Christianson syndrome: - Genetic Testing Registry: Christianson syndrome - MedlinePlus Encyclopedia: Seizures These resources from MedlinePlus offer information about the diagnosis a...
4. `medquad_ans_0000868-3` (score=0.0159) - Renpenning syndrome is caused by mutations in the PQBP1 gene. This gene provides instructions for making a protein called polyglutamine-binding protein 1. This protein attaches (binds) to stretches of multiple copies of a protein buildin...
5. `medquad_ans_0000009-5` (score=0.0159) - These resources address the diagnosis or management of acral peeling skin syndrome: - Birmingham Children's Hospital, National Health Service (UK) - Genetic Testing Registry: Peeling skin syndrome, acral type These resources from Medline...

**Answer**:

The provided context does not specify any treatments for Renpenning syndrome. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and genetic testing, it does not detail specific treatments for Renpenning syndrome [medquad_ans_0000868-5]. Therefore, based on the available information, no specific treatments can be identified.

**Citations detected**: `medquad_ans_0000868-5`
**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

## Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Top docs**:

1. `medquad_ans_0001106-1` (score=0.0298) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2H? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2H. If the information is available, the table be...
2. `medquad_ans_0001108-1` (score=0.0291) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2J? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2J. If the information is available, the table be...
3. `medquad_ans_0001110-1` (score=0.0286) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is available, the table be...
4. `medquad_ans_0001112-1` (score=0.0284) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is available, the table be...
5. `medquad_ans_0001100-1__dup1` (score=0.0280) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is available, the table ...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001106-1] and [medquad_ans_0001108-1] discuss symptoms for types 2H and 2J, respectively, but there is no mention of type 2G. Therefore, the symptoms of Charcot-Marie-Tooth disease type 2G cannot be determined from the given context.

**Citations detected**: `medquad_ans_0001106-1`, `medquad_ans_0001108-1`
**Supported citations**: `medquad_ans_0001106-1`, `medquad_ans_0001108-1`
**Unsupported citations**: _none_

## Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Top docs**:

1. `medquad_ans_0001393-1` (score=0.0164) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
2. `medquad_ans_0001393-2` (score=0.0164) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure when standing or walki...
3. `medquad_ans_0001393-4` (score=0.0161) - What treatment is available for coccygodynia? Treatment for coccygodynia generally falls into conservative management or surgical intervention categories. The conservative approach typically includes hot sitz baths, NSAIDs, stool softene...
4. `medquad_ans_0001807-1` (score=0.0161) - What are the signs and symptoms of Desmoid tumor? The Human Phenotype Ontology provides the following list of signs and symptoms for Desmoid tumor. If the information is available, the table below includes how often the symptom is seen i...
5. `medquad_ans_0000108-2__dup1` (score=0.0159) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Citations detected**: `medquad_ans_0001393-2`
**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

## Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.0328) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgenesis and cleft lip-p...
2. `medquad_ans_0001811-1` (score=0.0301) - What are the signs and symptoms of Desmosterolosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Desmosterolosis. If the information is available, the table below includes how often the symptom is se...
3. `medquad_ans_0004026-1` (score=0.0290) - What are the signs and symptoms of Microcephaly, holoprosencephaly, and intrauterine growth retardation? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, holoprosencephaly, and intrauterine...
4. `medquad_ans_0004592-1` (score=0.0286) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table below includes how o...
5. `medquad_ans_0006607-2` (score=0.0277) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, the table below includ...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are reported in the context provided, with frequencies estimated based on studies. For example, cleft palate and cleft upper lip are noted as occurring in individuals with this condition, along with hypoplasia of the corpus callosum and microcephaly [medquad_ans_0004025-1]. Specific frequencies are not provided for all symptoms in this context.

**Citations detected**: `medquad_ans_0004025-1`
**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

## Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.0328) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal a...
2. `medquad_ans_0003402-2` (score=0.0313) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenital). The 3 most commo...
3. `medquad_ans_0003402-6` (score=0.0313) - How might Klippel-Feil syndrome be treated? There is no cure for Klippel Feil syndrome (KFS); treatment is generally symptomatic and supportive. Management depends on the features and severity in each person, and can be life-long. Carefu...
4. `medquad_ans_0000562-5` (score=0.0303) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing Registry: Klippel-F...
5. `medquad_ans_0005865-2` (score=0.0292) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may not have noticeable...

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are required to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out heart and lung involvement, examination of the chest wall for rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present [medquad_ans_0003402-5].

**Citations detected**: `medquad_ans_0003402-5`
**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

## Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0000222-1` (score=0.0323) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of the diaphragm before ...
2. `medquad_ans_0000222-2` (score=0.0313) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
3. `medquad_ans_0000222-4` (score=0.0313) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...
4. `medquad_ans_0001497-2` (score=0.0308) - What are the signs and symptoms of Congenital diaphragmatic hernia? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital diaphragmatic hernia. If the information is available, the table below incl...
5. `medquad_ans_0001497-1` (score=0.0302) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severity from a thinned are...

**Answer**:

Congenital diaphragmatic hernia is a defect in the diaphragm resulting from abnormal development before birth. The diaphragm, composed of muscle and fibrous tissue, separates the abdominal and chest cavities. A defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs, which can lead to pulmonary hypoplasia and life-threatening breathing difficulties at birth. This condition ranges from a thinned area in the diaphragm to its complete absence. It is classified by position: Bochdalek hernias (80–90% of cases, involving the side or back of the diaphragm), Morgagni hernias (about 2% of cases, involving the front part, less likely to cause severe symptoms), and rarer types affecting the central region or involving only a thin membrane. Approximately 1 in 2,500 newborns is affected, and in

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Top docs**:

1. `medquad_ans_0000173-1` (score=0.0328) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as touch, pain, heat, and ...
2. `medquad_ans_0001088-1__dup1` (score=0.0320) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common symptoms may includ...
3. `medquad_ans_0001094-1__dup1` (score=0.0276) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent between age 5 and 25 years...
4. `medquad_ans_0001105-1` (score=0.0263) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is available, the table be...
5. `medquad_ans_0006495-1` (score=0.0260) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 4. If the information is availab...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Damage to these nerves can result in loss of sensation and wasting (atrophy) of muscles in the feet, legs, and hands. The disease typically becomes apparent in adolescence or early adulthood, though onset can occur from early childhood through late adulthood. Symptoms vary in severity even among family members and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches (pes cavus), flat feet (pes planus), or curled toes (hammer toes), difficulty flexing the foot or walking on the heel, a higher than normal step, and decreased sensitivity to touch, heat, and cold. In some cases, individuals may experience gradual hearing loss, deafness, or loss of vision. There are

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Top docs**:

1. `medquad_ans_0000669-4` (score=0.0323) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suf...
2. `medquad_ans_0000669-2` (score=0.0164) - The prevalence of monilethrix is unknown.
3. `medquad_ans_0000669-1` (score=0.0161) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin word for necklace (m...
4. `medquad_ans_0000785-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000577-4` (score=0.0159) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

Yes, monilethrix can be inherited in different patterns depending on the cause. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. In rare cases, monilethrix results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000669-4]. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene but typically do not show signs and symptoms [medquad_ans_0000669-4].

**Citations detected**: `medquad_ans_0000669-4`
**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

## Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Top docs**:

1. `medquad_ans_0000215-4__dup2` (score=0.0318) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1
2. `medquad_ans_0000223-12` (score=0.0308) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, the part of the dige...
3. `medquad_ans_0000215-3__dup2` (score=0.0304) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the ureters attach to the bl...
4. `medquad_ans_0000217-4__dup1` (score=0.0301) - In many cases, a child with VUR has no symptoms. When symptoms are present, the most common is a urinary tract infection (UTI). VUR can lead to infection because urine that remains in the childs urinary tract provides a place for bacteri...
5. `medquad_ans_0000223-2__dup2` (score=0.0299) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has several systems to pre...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microbes, most commonly bacteria that live in the bowel, such as Escherichia coli (E. coli). These bacteria can attach to the lining of the urinary tract and cause infection despite the urinary tract's natural defenses, including one-way valves that prevent urine from backing up and immune defenses that help prevent infection. Most UTIs are not serious, but some can lead to serious problems. Symptoms may range from mild to severe or may be absent. A UTI is diagnosed by testing a urine sample and treated with antibiotics. Children who delay urination, have inadequate fluid intake, or suffer from chronic constipation are at higher risk due to increased bacterial growth. After treatment, a healthcare provider may recommend further tests to check for conditions like vesicoureteral reflux (VUR) or urinary tract blockages [medquad_ans_000021

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000804-1` (score=0.0325) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the blood. The growth o...
2. `medquad_ans_0004939-2` (score=0.0318) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their ability to filter waste pr...
3. `medquad_ans_0000135-2__dup2` (score=0.0315) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems with blood vessels in ...
4. `medquad_ans_0000587-2__dup1` (score=0.0292) - What are the signs and symptoms of Autosomal recessive polycystic kidney disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive polycystic kidney disease. If the information is ava...
5. `medquad_ans_0000167-18` (score=0.0292) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common form of PKD and the mo...

**Answer**:

The complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones, heart valve abnormalities, and an increased risk of an abnormal bulging (aneurysm) in the aorta or in blood vessels at the base of the brain. Aneurysms can be life-threatening if they tear or rupture [medquad_ans_0000804-1].

**Citations detected**: `medquad_ans_0000804-1`
**Supported citations**: `medquad_ans_0000804-1`
**Unsupported citations**: _none_

## Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.0325) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contaminated food or water....
2. `medquad_ans_0000123-4__dup2` (score=0.0320) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 months old. Symptoms u...
3. `medquad_ans_0000123-6` (score=0.0318) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care provider may ask for...
4. `medquad_ans_0000391-1__dup1` (score=0.0317) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. Viral gastroenteritis...
5. `medquad_ans_0001853-3` (score=0.0154) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_0000649-

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0001497-3` (score=0.0311) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic set of findings. Curre...
2. `medquad_ans_0000222-4` (score=0.0306) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...
3. `medquad_ans_0000222-3` (score=0.0164) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow syndrome, Fryns syn...
4. `medquad_ans_0000222-1` (score=0.0161) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of the diaphragm before ...
5. `medquad_ans_0000538-3` (score=0.0161) - Changes in the DNA near the SOX9 gene are the most common genetic cause of isolated Pierre Robin sequence. It is likely that changes in other genes, some of which have not been identified, also cause isolated Pierre Robin sequence. The S...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia (CDH) include:

- Recognized chromosome abnormalities that may cause syndromic CDH [medquad_ans_0001497-3].
- Single gene disorders that result in CDH [medquad_ans_0001497-3].
- Chromosomal microdeletions (tiny loss of genetic material) or microduplications (extra copy of genetic material) that may be undetectable currently [medquad_ans_0001497-3].
- Mutations in major genes important for diaphragm development [medquad_ans_0001497-3].
- Combined effects of multiple minor genetic mutations or variants, suggesting a polygenic inheritance pattern [medquad_ans_0001497-3].
- Gene-environment interactions (multifactorial

**Citations detected**: `medquad_ans_0001497-3`
**Supported citations**: `medquad_ans_0001497-3`
**Unsupported citations**: _none_

## Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.0323) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 and RAG2 genes have ...
2. `medquad_ans_0004534-2` (score=0.0320) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCID). The characterist...
3. `medquad_ans_0004534-1` (score=0.0320) - Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) characterized by erythroderma (skin redness), desquamation (peeling skin), alopecia (hair loss), chronic diarrhea, failure to thrive, lymphadenopath...
4. `medquad_ans_0004534-4` (score=0.0318) - How might Omenn syndrome be treated? The standard treatment for Omenn syndrome is bone marrow transplantation or cord blood stem cell transplantation. General care for any patient with severe combined immunodeficiency (SCID), including O...
5. `medquad_ans_0001853-3` (score=0.0154) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG1 and RAG2 genes located on chromosome band 11p13. Mutations in other genes such as ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4 have also been associated with Omenn syndrome. Additionally, some cases are found in association with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Citations detected**: `medquad_ans_0004534-3`
**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

## Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Top docs**:

1. `medquad_ans_0000097-5` (score=0.0164) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 - Johns Hopkins Medic...
2. `medquad_ans_0000785-4` (score=0.0164) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000097-1` (score=0.0161) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and...
4. `medquad_ans_0000626-4` (score=0.0161) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0005792-2` (score=0.0159) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive 7. If the information is avail...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and symptoms of the disorder first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive, often resulting in the need for a cane, walker, or wheelchair. This condition is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations, and each parent of an affected individual carries one copy of the mutated gene without

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.0318) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
2. `medquad_ans_0000050-2__dup1` (score=0.0318) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
3. `medquad_ans_0000123-1__dup2` (score=0.0317) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed arm - Lack of muscl...
4. `medquad_ans_0000049-4__dup1` (score=0.0299) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
5. `medquad_ans_0000050-4__dup1` (score=0.0299) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.

**Answer**:

Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery [medquad_ans_0000049-2__dup1, medquad_ans_0000050-2__dup1, medquad_ans_0000123-1__dup2].

**Citations detected**: `medquad_ans_0000049-2__dup1, medquad_ans_0000050-2__dup1, medquad_ans_0000123-1__dup2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000049-2__dup1, medquad_ans_0000050-2__dup1, medquad_ans_0000123-1__dup2`

## Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Top docs**:

1. `medquad_ans_0000071-11` (score=0.0318) - Although everyone has some risk for UTIs, some people are more likely to get UTIs than others. These include people who have - spinal cord injuries or other nerve damage around the bladder - a blockage in the urinary tract that can trap ...
2. `medquad_ans_0000216-4__dup1` (score=0.0317) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ultrasound of a baby...
3. `medquad_ans_0000216-7` (score=0.0313) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
4. `medquad_ans_0000216-8` (score=0.0305) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PUV) - ureterocele - S...
5. `medquad_ans_0000216-6` (score=0.0303) - Treatment for urine blockage depends on the cause and severity of the blockage. Hydronephrosis discovered before the baby is born rarely requires immediate action, especially if it is only on one side. The condition often goes away witho...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, such as vesicoureteral reflux [medquad_ans_0000216-4__dup1]. These defects can lead to a blockage that prevents the normal flow of urine, causing urine to back up and result in conditions like hydroureter and hydronephrosis [medquad_ans_0000216-4__dup1]. However, researchers have not found that a mother's eating, diet, or nutrition plays a role in causing or preventing urine blockage in newborns [medquad_ans_0000216-7].

**Citations detected**: `medquad_ans_0000216-4__dup1`, `medquad_ans_0000216-7`
**Supported citations**: `medquad_ans_0000216-4__dup1`, `medquad_ans_0000216-7`
**Unsupported citations**: _none_

## Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Top docs**:

1. `medquad_ans_0000526-2` (score=0.0312) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
2. `medquad_ans_0000526-4` (score=0.0267) - Intranuclear rod myopathy is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases are not inherited; they result from new mutations in the gene and occur i...
3. `medquad_ans_0000526-1` (score=0.0164) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor muscle tone (hypoto...
4. `medquad_ans_0000989-2` (score=0.0164) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000526-3` (score=0.0161) - Intranuclear rod myopathy is caused by a mutation in the ACTA1 gene. This gene provides instructions for making a protein called skeletal alpha ()-actin, which is part of the actin protein family. Actin proteins are important for cell mo...

**Answer**:

Intranuclear rod myopathy affects an estimated 1 in 50,000 people [medquad_ans_0000989-2].

**Citations detected**: `medquad_ans_0000989-2`
**Supported citations**: `medquad_ans_0000989-2`
**Unsupported citations**: _none_

## Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Top docs**:

1. `medquad_ans_0000036-1__dup3` (score=0.0320) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. What Is Blood Pressure? Blood pressure is the force of blood pushing against the walls of the blood vessels as...
2. `medquad_ans_0000036-11` (score=0.0315) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure. Primary, or essent...
3. `medquad_ans_0000150-8` (score=0.0315) - - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood vessels as it mov...
4. `medquad_ans_0000071-1__dup2` (score=0.0299) - Espaol High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. Measuring Blood Pressure Blood pressure is the force of blood pushing against the walls of the arteries...
5. `medquad_ans_0000471-1__dup1` (score=0.0296) - Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blood. This is called s...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. It is defined as a condition where blood pressure rises and stays high over time, resulting in increased force of blood flow against the walls of blood vessels. This force can make the heart work too hard and can harm arteries and organs such as the heart, kidneys, brain, and eyes [medquad_ans_0000036-1__dup3]. There are two main types: primary (or essential) high blood pressure, which is the most common and develops over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or the use of certain medicines and usually resolves when the underlying cause is treated or removed [medquad_ans_0000036-1__dup3]. Blood pressure is measured as two numbers—the systolic pressure (when the heart beats)

**Citations detected**: `medquad_ans_0000036-1__dup3`
**Supported citations**: `medquad_ans_0000036-1__dup3`
**Unsupported citations**: _none_

## Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Top docs**:

1. `medquad_ans_0001051-1` (score=0.0164) - X-linked dystonia-parkinsonism is a movement disorder that has been found only in people of Filipino descent. This condition affects men much more often than women. Parkinsonism is usually the first sign of X-linked dystonia-parkinsonism...
2. `medquad_ans_0001051-4` (score=0.0164) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
3. `medquad_ans_0001051-2` (score=0.0161) - X-linked dystonia-parkinsonism has been reported in more than 500 people of Filipino descent, although it is likely that many more Filipinos are affected. Most people with this condition can trace their mother's ancestry to the island of...
4. `medquad_ans_0000342-4` (score=0.0161) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0001051-5` (score=0.0159) - These resources address the diagnosis or management of X-linked dystonia-parkinsonism: - Gene Review: Gene Review: X-Linked Dystonia-Parkinsonism Syndrome - Genetic Testing Registry: Dystonia 3, torsion, X-linked These resources from Med...

**Answer**:

Yes, X-linked dystonia-parkinsonism is inherited in an X-linked recessive pattern [medquad_ans_0001051-4].

**Citations detected**: `medquad_ans_0001051-4`
**Supported citations**: `medquad_ans_0001051-4`
**Unsupported citations**: _none_

## Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Top docs**:

1. `medquad_ans_0000120-2` (score=0.0328) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
2. `medquad_ans_0000120-1` (score=0.0320) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molec...
3. `medquad_ans_0000702-1__dup1` (score=0.0315) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are mole...
4. `medquad_ans_0000120-3` (score=0.0313) - Mutations in the ACAT1 gene cause beta-ketothiolase deficiency. This gene provides instructions for making an enzyme that is found in the energy-producing centers within cells (mitochondria). This enzyme plays an essential role in breaki...
5. `medquad_ans_0000120-5` (score=0.0308) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer information about the ...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene, which provides instructions for making an enzyme found in the mitochondria. This enzyme plays an essential role in breaking down proteins and fats from the diet. Mutations in the ACAT

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Top docs**:

1. `medquad_ans_0000006_5-1` (score=0.0320) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - There are two types o...
2. `medquad_ans_0000006_5-7` (score=0.0318) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread...
3. `medquad_ans_0000006_5-4` (score=0.0317) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all through the brain stem. ...
4. `medquad_ans_0000006_5-9` (score=0.0311) - Key Points - There are different types of treatment for children with brain stem glioma. - Children with brain stem glioma should have their treatment planned by a team of health care providers who are experts in treating childhood brain...
5. `medquad_ans_0000006_5-5` (score=0.0296) - Tests that examine the brain are used to detect (find) childhood brain stem glioma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of health, including checking fo...

**Answer**:

The provided context does not mention any stages of childhood brain stem glioma. [medquad_ans_0000006_5-1] discusses types (such as diffuse intrinsic pontine glioma [DIPG] and focal glioma), characteristics, causes, symptoms, diagnosis, and treatment, but it does not outline stages of the disease. Therefore, the stages of childhood brain stem glioma cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000006_5-1`, `DIPG`
**Supported citations**: `medquad_ans_0000006_5-1`
**Unsupported citations**: `DIPG`

## Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.0328) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal cord (conus medullari...
2. `medquad_ans_0000266-3__dup1` (score=0.0320) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears to improve chances fo...
3. `medquad_ans_0004257-2` (score=0.0320) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms of this condition i...
4. `medquad_ans_0000156-1__dup1` (score=0.0286) - Hydromyelia refers to an abnormal widening of the central canal of the spinal cord that creates a cavity in which cerebrospinal fluid (commonly known as spinal fluid) can accumulate. As spinal fluid builds up, it may put abnormal pressur...
5. `medquad_ans_0000933-1` (score=0.0285) - Spina bifida is a condition in which the neural tube, a layer of cells that ultimately develops into the brain and spinal cord, fails to close completely during the first few weeks of embryonic development. As a result, when the spine fo...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. Symptoms in children may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to chronic strain on the spinal cord, which may be exacerbated

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Top docs**:

1. `medquad_ans_0000419-1__dup1` (score=0.0298) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel syndrome - compress...
2. `medquad_ans_0000155-1__dup3` (score=0.0257) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? It could be carpal t...
3. `medquad_ans_0000058-4__dup1` (score=0.0253) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to conduct fundamental research on the brain and nervous system, and to use that knowledge to reduce the burden of neurological disease. NINDS-funded s...
4. `medquad_ans_0000058-3__dup1` (score=0.0230) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work, people can perform ...
5. `medquad_ans_0000043_1-4` (score=0.0164) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The patient's age and general health. - Whether t...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as it responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Early diagnosis and treatment are important to prevent permanent nerve damage [medquad_ans_0000155-1__dup3].

**Citations detected**: `medquad_ans_0000058-3__dup1`, `medquad_ans_0000155-1__dup3`
**Supported citations**: `medquad_ans_0000058-3__dup1`, `medquad_ans_0000155-1__dup3`
**Unsupported citations**: _none_

## Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Top docs**:

1. `medquad_ans_0000296-5` (score=0.0318) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphragmatic Hernia - Medli...
2. `medquad_ans_0000296-3` (score=0.0164) - Mutations in the LRP2 gene cause Donnai-Barrow syndrome. The LRP2 gene provides instructions for making a protein called megalin, which functions as a receptor. Receptor proteins have specific sites into which certain other proteins, cal...
3. `medquad_ans_0000296-1` (score=0.0161) - Donnai-Barrow syndrome is an inherited disorder that affects many parts of the body. This disorder is characterized by unusual facial features, including prominent, wide-set eyes with outer corners that point downward; a short bulbous no...
4. `medquad_ans_0000130-2__dup1` (score=0.0161) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportive care.
5. `medquad_ans_0003103-4` (score=0.0159) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moisturize) the skin, h...

**Answer**:

The provided context does not specify treatments for Donnai-Barrow syndrome. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and information on conditions like hearing loss and diaphragmatic hernia, it does not detail specific treatments for Donnai-Barrow syndrome itself [medquad_ans_0000296-5]. Therefore, based on the available information, the treatments for Donnai-Barrow syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000296-5`
**Supported citations**: `medquad_ans_0000296-5`
**Unsupported citations**: _none_

## Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Top docs**:

1. `medquad_ans_0000189-10` (score=0.0312) - - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. - The exa...
2. `medquad_ans_0000189-8` (score=0.0296) - Treatment for porphyria depends on the type of porphyria the person has and the severity of the symptoms. Acute Porphyrias A health care provider treats acute porphyrias with heme or glucose loading to decrease the livers production of p...
3. `medquad_ans_0000189-3__dup2` (score=0.0293) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences: - Acute porphyrias ...
4. `medquad_ans_0001134-1` (score=0.0292) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not conform to any of the reco...
5. `medquad_ans_0000729-1__dup1` (score=0.0287) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. There are two main typ...

**Answer**:

Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, from one parent. Porphyria cutanea tarda is usually an acquired disorder, meaning factors other than genes cause the enzyme deficiency [medquad_ans_0000189-10].

**Citations detected**: `medquad_ans_0000189-10`
**Supported citations**: `medquad_ans_0000189-10`
**Unsupported citations**: _none_

## Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.0328) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable degrees of hearing lo...
2. `medquad_ans_0001900-1` (score=0.0291) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual loss and color visi...
3. `medquad_ans_0000585-2__dup1` (score=0.0161) - What are the signs and symptoms of Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive optic atrophy, hearing ...
4. `medquad_ans_0006496-1` (score=0.0159) - X-linked Charcot-Marie-Tooth disease type 5 (CMTX5) is a neurological condition characterized by peripheral neuropathy, early-onset bilateral profound sensorineural hearing loss, and optic neuropathy leading to visual impairment. Periphe...
5. `medquad_ans_0000785-4` (score=0.0159) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is inherited in an autosomal recessive manner [medquad_ans_0000585-1__dup1].

**Citations detected**: `medquad_ans_0000585-1__dup1`
**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

## Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.0323) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process in cells known as g...
2. `medquad_ans_0000794-2` (score=0.0315) - Phosphoglycerate kinase deficiency appears to be a rare disorder. About 30 families with affected members have been reported in the scientific literature.
3. `medquad_ans_0000652-3` (score=0.0303) - Mutations in the MVK gene cause mevalonate kinase deficiency. The MVK gene provides instructions for making the mevalonate kinase enzyme. This enzyme is involved in the production of cholesterol, which is later converted into steroid hor...
4. `medquad_ans_0000794-4` (score=0.0298) - This condition is inherited in an X-linked recessive pattern. The PGK1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the gene in each cell is ...
5. `medquad_ans_0000795-3` (score=0.0290) - Phosphoglycerate mutase deficiency is caused by mutations in the PGAM2 gene. This gene provides instructions for making an enzyme called phosphoglycerate mutase, which is involved in a critical energy-producing process in cells known as ...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene [medquad_ans_0000794-3]. These mutations reduce the activity of the phosphoglycerate kinase enzyme, which disrupts energy production during glycolysis and leads to cell damage or death [medquad_ans_0000794-3]. It is unclear why this abnormality preferentially affects certain cell types, such as red blood cells, brain cells, or muscle cells, and researchers speculate that different mutations may have varying effects on enzyme activity in different cell types [medquad_ans_0000794-3].

**Citations detected**: `medquad_ans_0000794-3`
**Supported citations**: `medquad_ans_0000794-3`
**Unsupported citations**: _none_

## Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Top docs**:

1. `medquad_ans_0000849-3` (score=0.0313) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in this gene promotes cell...
2. `medquad_ans_0000109-4__dup2` (score=0.0297) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develop if: The walls of t...
3. `medquad_ans_0000608-3` (score=0.0257) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to form a tumor. In nearl...
4. `medquad_ans_0000850-5` (score=0.0164) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagnosis and management o...
5. `medquad_ans_0000453-1` (score=0.0161) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive disease is a condit...

**Answer**:

The provided context does not mention genetic changes related to pulmonary veno-occlusive disease. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000849-3, medquad_ans_0000109-4__dup2]

**Citations detected**: `medquad_ans_0000849-3, medquad_ans_0000109-4__dup2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000849-3, medquad_ans_0000109-4__dup2`

## Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Top docs**:

1. `medquad_ans_0001006-1` (score=0.0328) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up in tissues and organs...
2. `medquad_ans_0006260-1` (score=0.0305) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate hydrolase, one of the...
3. `medquad_ans_0006262-1` (score=0.0253) - Tyrosinemia type 3 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a deficiency of the enzyme 4-hydroxyphenylpyruvate dioxygenase, on...
4. `medquad_ans_0001006-3` (score=0.0252) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by the kidneys or used t...
5. `medquad_ans_0001051-3` (score=0.0250) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells and tissues throughou...

**Answer**:

The genetic changes related to tyrosinemia include mutations in the FAH gene, which cause a shortage of the enzyme fumarylacetoacetate hydrolase. This enzyme is necessary for the multi-step process that breaks down tyrosine, and its deficiency leads to the accumulation of tyrosine and its byproducts [medquad_ans_0001006-1]. Tyrosinemia type I, the most severe form, is specifically caused by these mutations in the FAH gene [medquad_ans_0001006-1].

**Citations detected**: `medquad_ans_0001006-1`
**Supported citations**: `medquad_ans_0001006-1`
**Unsupported citations**: _none_

## Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.0278) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 - Genetic Testing Reg...
2. `medquad_ans_0002675-2` (score=0.0164) - What are the signs and symptoms of Graves' disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Graves' disease. If the information is available, the table below includes how often the symptom is se...
3. `medquad_ans_0000004-6__dup2` (score=0.0164) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease is often diagnosed a...
4. `medquad_ans_0000108-2__dup1` (score=0.0161) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
5. `medquad_ans_0000004-7__dup2` (score=0.0161) - Treatment for Graves disease can sometimes affect pregnancy. After treatment with surgery or radioactive iodine, TSI antibodies can still be present in the blood, even when thyroid levels are normal. If a pregnant woman has received eith...

**Answer**:

The treatments for Graves' Disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves' disease in the United States. [medquad_ans_0002675-2]

**Citations detected**: `medquad_ans_0002675-2`
**Supported citations**: `medquad_ans_0002675-2`
**Unsupported citations**: _none_

## Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.0325) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, or stumbling. Sympto...
2. `medquad_ans_0000545-1` (score=0.0323) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor disability, speech ...
3. `medquad_ans_0000039-3__dup1` (score=0.0318) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten disease is often fatal b...
4. `medquad_ans_0000545-3` (score=0.0310) - Most cases of juvenile Batten disease are caused by mutations in the CLN3 gene. This gene provides instructions for making a protein whose function is unknown. It is unclear how mutations in the CLN3 gene lead to the characteristic featu...
5. `medquad_ans_0000039-2__dup1` (score=0.0304) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be treated appropriately a...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. The early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling. Vision loss is often an early sign and may lead to suspicion during an eye exam. Symptoms progress over time and include progressive vision loss, intellectual and motor disability, speech difficulties, and seizures. In juvenile Batten disease, vision impairment typically begins between ages 4 and 8, followed by rapid progression to blindness and developmental regression, including loss of speech and motor skills such as walking or sitting. Affected individuals also experience cognitive impairment, worsening seizures, and loss of previously acquired skills. The disorder is part of a group of conditions known as neuronal ceroid lipofuscinoses (NCLs), which affect the nervous system and

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Top docs**:

1. `medquad_ans_0002061-1` (score=0.0323) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, hair, and skeletal s...
2. `medquad_ans_0002061-5` (score=0.0320) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak muscle tone and delayed ...
3. `medquad_ans_0002061-3` (score=0.0318) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This gene provides instruc...
4. `medquad_ans_0002061-2` (score=0.0315) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and delayed healing of sk...
5. `medquad_ans_0002061-4` (score=0.0305) - How is Ehlers-Danlos syndrome progeroid type inherited? Ehlers-Danlos syndrome progeroid type is inherited in an autosomal recessive pattern. This means that an individual must have two non-functional copies of the B4GALT7 gene to be aff...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. It is characterized by unusually flexible joints, loose elastic skin, and easy scarring. Unique features include sparse scalp hair and eyebrows, and loose elastic skin on the face, which cause affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in affected children. The disorder is caused by mutations in both copies of the B4GALT7 gene, located on chromosome 5, which leads to reduced production of an enzyme involved in collagen formation, resulting in improperly formed collagen and weak connective tissue [medquad_ans_0002061-1, medquad_ans_0002061-3]. It is inherited in an autosomal recessive pattern [medquad_ans_0

**Citations detected**: `medquad_ans_0002061-1, medquad_ans_0002061-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0002061-1, medquad_ans_0002061-3`

## Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.0328) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
2. `medquad_ans_0000416-2` (score=0.0310) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
3. `medquad_ans_0002601-1` (score=0.0301) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form have minor delays i...
4. `medquad_ans_0000416-1` (score=0.0161) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with the mild form of glu...
5. `medquad_ans_0000442-3` (score=0.0161) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step production (synthesis)...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3]. The specific relationship between these genetic changes and the severity of the condition (mild or severe) is unclear [medquad_ans_0000416-3].

**Citations detected**: `medquad_ans_0000416-3`
**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

## Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.0320) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. If the information is...
2. `medquad_ans_0001047-1` (score=0.0315) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) near the ends of bones ...
3. `medquad_ans_0001048-1` (score=0.0305) - X-linked chondrodysplasia punctata 2 is a disorder characterized by bone, skin, and eye abnormalities. It occurs almost exclusively in females. Although the signs and symptoms of this condition vary widely, almost all affected individual...
4. `medquad_ans_0005365-1` (score=0.0294) - What are the signs and symptoms of Rhizomelic chondrodysplasia punctata type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Rhizomelic chondrodysplasia punctata type 3. If the information is availab...
5. `medquad_ans_0001170-2` (score=0.0293) - What are the signs and symptoms of Chondrodysplasia punctata 2 X-linked dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 2 X-linked dominant. If the information is ava...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2]. Additionally, characteristic features include short stature, unusually short fingertips and ends of the toes, and a flattened nose with crescent-shaped nostrils and a flat nasal bridge [medquad_ans_0001047-1]. Epiphyseal stippling is typically observed in the bones of the ankles, toes, and fingers in infancy and may disappear in early childhood [medquad_ans_0001047-1

**Citations detected**: `medquad_ans_0001169-2`, `medquad_ans_0001047-1`
**Supported citations**: `medquad_ans_0001169-2`, `medquad_ans_0001047-1`
**Unsupported citations**: _none_

## Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Top docs**:

1. `medquad_ans_0000132-11` (score=0.0297) - - Whipple disease is a rare bacterial infection that primarily affects the small intestine. Left untreated, Whipple disease gets worse and is usually life threatening. - Bacteria called Tropheryma whipplei (T. whipplei) cause Whipple dis...
2. `medquad_ans_0000132-6` (score=0.0273) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following complications in other ar...
3. `medquad_ans_0000025_1-7` (score=0.0230) - Key Points - There are different types of treatment for patients with pancreatic cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Chemoradiation therapy - Targeted therapy - There are tr...
4. `medquad_ans_0000132-10` (score=0.0164) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
5. `medquad_ans_0000132-5__dup1` (score=0.0161) - Signs and symptoms of Whipple disease can vary widely from person to person. The most common symptoms of Whipple disease are - diarrhea - weight loss caused by malabsorption A person may not have diarrhea. Instead, other signs and sympto...

**Answer**:

The treatments for Whipple disease include intravenous (IV) antibiotics for the first 2 weeks of treatment, followed by long-term oral antibiotics. Most patients have good outcomes with an early diagnosis and complete treatment [medquad_ans_0000132-11].

**Citations detected**: `medquad_ans_0000132-11`
**Supported citations**: `medquad_ans_0000132-11`
**Unsupported citations**: _none_

## Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.0313) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-1__dup1` (score=0.0308) - Bell's palsy is a form of temporary facial paralysis which results from damage or trauma to one of the facial nerves. This disorder is characterized by the sudden onset of facial paralysis that often affects just one side and can cause s...
3. `medquad_ans_0000042-3__dup2` (score=0.0306) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks after the initial onset ...
4. `medquad_ans_0000086-1__dup2` (score=0.0298) - Bell's palsy is the most common cause of facial paralysis. It usually affects just one side of the face. Symptoms appear suddenly and are at their worst about 48 hours after they start. They can range from mild to severe and include - Tw...
5. `medquad_ans_0000042-2__dup2` (score=0.0271) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminop...

**Answer**:

Treatments for Bell's palsy include steroids such as prednisone, which reduce inflammation and swelling of the facial nerve [medquad_ans_0000676-1\_\_dup1][medquad_ans_0000042-1__dup2]. Antiviral drugs like acyclovir may have some benefit in shortening the course of the disease [medquad_ans_0000042-1__dup2]. Analgesics such as aspirin, acetaminophen, or ibuprofen can relieve pain [medquad_ans_0000676-1\_\_dup1][medquad_ans_0000042-1__dup2]. Additional therapies that have been used include physical therapy, facial massage, and acupuncture [medquad_ans_0000676-1__dup1]medquad_ans_0

**Citations detected**: `medquad_ans_0000676-1__dup1`, `medquad_ans_0000042-1__dup2`
**Supported citations**: `medquad_ans_0000676-1__dup1`
**Unsupported citations**: `medquad_ans_0000042-1__dup2`
