# Baseline Generation Examples (No Reranker)

- retriever: `bi_encoder`
- llm_model: `qwen/qwen3-4b-instruct-2507`
- top_k: `5`
- reranker_enabled: `False`
- num_examples: `300`

## Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Top docs**:

1. `medquad_ans_0001059-5` (score=0.6721) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndrome 1, X-linked - Ge...
2. `medquad_ans_0001059-1` (score=0.6546) - X-linked lymphoproliferative disease (XLP) is a disorder of the immune system and blood-forming cells that is found almost exclusively in males. More than half of individuals with this disorder experience an exaggerated immune response t...
3. `medquad_ans_0000483-2` (score=0.6533) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
4. `medquad_ans_0001052-5` (score=0.6504) - These resources address the diagnosis or management of X-linked hyper IgM syndrome: - Gene Review: Gene Review: X-Linked Hyper IgM Syndrome - Genetic Testing Registry: Immunodeficiency with hyper IgM type 1 - MedlinePlus Encyclopedia: Im...
5. `medquad_ans_0000814-2` (score=0.6454) - The prevalence of PPM-X syndrome is unknown.

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 (XLP1) include an exaggerated immune response to the Epstein-Barr virus (EBV), leading to the production of abnormally large numbers of T cells, B cells, and macrophages. This proliferation often causes hemophagocytic lymphohistiocytosis, which presents with fever and destruction of blood-producing cells in the bone marrow, damaging the liver and potentially affecting the spleen, heart, kidneys, and other organs. Additionally, about one-third of individuals with XLP have dysgammaglobulinemia, characterized by abnormal levels of antibodies, increasing susceptibility to recurrent infections. Lymphomas also occur in about one-third of individuals with XLP. Without treatment, most people with XLP survive only into childhood, and death usually results from hemophagocytic lymphohistiocytosis [medquad_ans_0001059-

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Top docs**:

1. `medquad_ans_0000312-2` (score=0.6889) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,000 people in this p...
2. `medquad_ans_0000859-2` (score=0.6596) - Rapid-onset dystonia parkinsonism appears to be a rare disorder, although its prevalence is unknown. It has been diagnosed in individuals and families from the United States, Europe, and Korea.
3. `medquad_ans_0000312-5` (score=0.6564) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Movement - uncontrolled...
4. `medquad_ans_0000112-1__dup1` (score=0.6405) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single muscle; a group of ...
5. `medquad_ans_0000312-4` (score=0.6375) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the disorder and may neve...

**Answer**:

Yes, early-onset primary dystonia is inherited. Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, and everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent [medquad_ans_0000312-4]. Although only 30 to 40 percent of people who inherit a TOR1A mutation will develop signs and symptoms, the condition is often inherited and appears to be linked to genetic factors [medquad_ans_0000312-4]. About half of all dystonia cases are primary or idiopathic, with many of these cases being inherited [medquad_ans_0000112-1__dup1].

**Citations detected**: `medquad_ans_0000312-4`, `medquad_ans_0000112-1__dup1`
**Supported citations**: `medquad_ans_0000312-4`, `medquad_ans_0000112-1__dup1`
**Unsupported citations**: _none_

## Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Top docs**:

1. `medquad_ans_0000121-7` (score=0.6644) - Some people with cysticercosis do not need to be treated. There are medications available to treat cysticercosis for those who do need treatment. Sometimes surgery may be needed. Your doctor will advise you on which treatment is best for...
2. `medquad_ans_0000397-2__dup1` (score=0.6555) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat raw or undercooked bee...
3. `medquad_ans_0000121-6` (score=0.6530) - If you think that you may have cysticercosis, please see your health care provider. Your health care provider will ask you about your symptoms, where you have travelled, and what kinds of foods you eat. The diagnosis of neurocysticercosi...
4. `medquad_ans_0000121-2__dup3` (score=0.6481) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there (these are called c...
5. `medquad_ans_0000397-5__dup1` (score=0.6374) - Diagnosis of Taenia tapeworm infections is made by examination of stool samples; individuals should also be asked if they have passed tapeworm segments. Stool specimens should be collected on three different days and examined in the lab ...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef or pork. Infections with Taenia saginata occur where contaminated raw beef is consumed, particularly in Eastern Europe, Russia, eastern Africa, and Latin America. Taenia solium infections are more common in underdeveloped communities with poor sanitation and where raw or undercooked pork is consumed, with higher rates observed in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia. Taenia solium taeniasis is also seen in the United States, typically among Latin American immigrants. Taenia asiatica is limited to Asia and is found in countries such as the Republic of Korea, China, Taiwan, Indonesia, and Thailand [medquad_ans_0000397-2__dup1].

**Citations detected**: `medquad_ans_0000397-2__dup1`
**Supported citations**: `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

## Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Top docs**:

1. `medquad_ans_0000029-4` (score=0.7188) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an au...
2. `medquad_ans_0000029-2` (score=0.7173) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
3. `medquad_ans_0000029-5` (score=0.6755) - These resources address the diagnosis or management of Aicardi-Goutieres syndrome: - Gene Review: Gene Review: Aicardi-Goutieres Syndrome - Genetic Testing Registry: Aicardi Goutieres syndrome - Genetic Testing Registry: Aicardi Goutiere...
4. `medquad_ans_0000028-2` (score=0.6724) - Aicardi syndrome is a very rare disorder. It occurs in about 1 in 105,000 to 167,000 newborns in the United States. Researchers estimate that there are approximately 4,000 affected individuals worldwide.
5. `medquad_ans_0000029-1` (score=0.6526) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. However, about 20 percent ...

**Answer**:

The provided context does not specify the symptoms of Aicardi-Goutieres syndrome type 2. While general symptoms of Aicardi-Goutieres syndrome are mentioned, such as enlarged liver and spleen, elevated liver enzymes, decreased blood platelets, and abnormal neurological responses, there is no information specifically detailing the symptoms associated with type 2 [medquad_ans_0000029-1]. Therefore, the symptoms of Aicardi-Goutieres syndrome type 2 cannot be determined from the given context.

**Citations detected**: `medquad_ans_0000029-1`
**Supported citations**: `medquad_ans_0000029-1`
**Unsupported citations**: _none_

## Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Top docs**:

1. `medquad_ans_0000553-2` (score=0.7103) - Kabuki syndrome occurs in approximately 1 in 32,000 newborns.
2. `medquad_ans_0000569-2` (score=0.6962) - Kuskokwim syndrome is extremely rare. It affects a small number of people from the Yup'ik Eskimo population in southwest Alaska.
3. `medquad_ans_0003421-1` (score=0.6934) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how often the symptom i...
4. `medquad_ans_0000565-2` (score=0.6920) - Knobloch syndrome is a rare condition. However, the exact prevalence of the condition is unknown.
5. `medquad_ans_0000127-2` (score=0.6859) - Bjrnstad syndrome is a rare condition, although its prevalence is unknown. It has been found in populations worldwide.

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1]. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology and Orphanet, with specific frequencies provided as rough percentages or fractions derived from studies [medquad_ans_0003421-1].

**Citations detected**: `medquad_ans_0003421-1`
**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

## Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Top docs**:

1. `medquad_ans_0000450-2` (score=0.7197) - The two major forms of hemophilia occur much more commonly in males than in females. Hemophilia A is the most common type of the condition; 1 in 4,000 to 1 in 5,000 males worldwide are born with this disorder. Hemophilia B occurs in appr...
2. `medquad_ans_0000102-1__dup3` (score=0.7045) - Normally, if you get hurt, your body forms a blood clot to stop the bleeding. For blood to clot, your body needs cells called platelets and proteins known as clotting factors. If you have a bleeding disorder, you either do not have enoug...
3. `medquad_ans_0000338-2` (score=0.7032) - Inherited factor XIII deficiency affects 1 to 3 per million people worldwide. Researchers suspect that mild factor XIII deficiency, including the acquired form of the disorder, is underdiagnosed because many affected people never have a ...
4. `medquad_ans_0000455-1__dup1` (score=0.6962) - Hemophilia is a rare disorder in which the blood does not clot normally. It is usually inherited. Hemophilia usually occurs in males. If you have hemophilia, you have little or no clotting factor. Clotting factor is a protein needed for ...
5. `medquad_ans_0001021-2` (score=0.6893) - Von Willebrand disease is estimated to affect 1 in 100 to 10,000 individuals. Because people with mild signs and symptoms may not come to medical attention, it is thought that this condition is underdiagnosed. Most researchers agree that...

**Answer**:

Yes, hemophilia is inherited. [medquad_ans_0000102-1__dup3] and [medquad_ans_0000455-1__dup1] state that hemophilia is an inherited bleeding disorder and usually occurs in males, indicating it is typically passed down through families.

**Citations detected**: `medquad_ans_0000102-1__dup3`, `medquad_ans_0000455-1__dup1`
**Supported citations**: `medquad_ans_0000102-1__dup3`, `medquad_ans_0000455-1__dup1`
**Unsupported citations**: _none_

## Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.6378) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often the symptom is seen i...
2. `medquad_ans_0000062-3__dup1` (score=0.6274) - Central pain syndrome is not a fatal disorder, but the syndrome causes disabling chronic pain and suffering among the majority of individuals who have it.
3. `medquad_ans_0000531-2` (score=0.6231) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
4. `medquad_ans_0000409-2` (score=0.6211) - Gitelman syndrome affects an estimated 1 in 40,000 people worldwide.
5. `medquad_ans_0000514-2` (score=0.6194) - IPEX syndrome is a rare disorder; its prevalence is unknown.

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Citations detected**: `medquad_ans_0003284-1`
**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

## Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Top docs**:

1. `medquad_ans_0000775-2` (score=0.7265) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
2. `medquad_ans_0000775-3` (score=0.6761) - Mutations in the PIGA gene cause paroxysmal nocturnal hemoglobinuria. The PIGA gene provides instructions for making a protein called phosphatidylinositol glycan class A. This protein takes part in a series of steps that produce a molecu...
3. `medquad_ans_0000775-5` (score=0.6573) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglobinuria - MedlinePlu...
4. `medquad_ans_0000476-2` (score=0.6503) - Hereditary spherocytosis occurs in 1 in 2,000 individuals of Northern European ancestry. This condition is the most common cause of inherited anemia in that population. The prevalence of hereditary spherocytosis in people of other ethnic...
5. `medquad_ans_0001062-2` (score=0.6502) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.

**Answer**:

Paroxysmal nocturnal hemoglobinuria is a rare disorder estimated to affect between 1 and 5 per million people. It is caused by somatic mutations in the PIGA gene, which occur in blood-forming hematopoietic stem cells in the bone marrow. These mutations result in the production of abnormal blood cells. The abnormal red blood cells lack two important complement-regulating proteins that normally protect them from destruction by the complement system, leading to their premature destruction and hemolytic anemia. Additionally, abnormal hematopoietic stem cells may be less susceptible to apoptosis, contributing to an increased proportion of abnormal blood cells. This can lead to symptoms such as hemoglobinuria and thrombosis [medquad_ans_0000775-2, medquad_ans_0000775-3].

**Citations detected**: `medquad_ans_0000775-2, medquad_ans_0000775-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000775-2, medquad_ans_0000775-3`

## Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.7064) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large in...
2. `medquad_ans_0000010-11` (score=0.7056) - Colorectal cancer accounts for almost ten percent of all cancer deaths in the United States. The risk of developing colorectal cancer rises after age 50. It is common in both men and women.
3. `medquad_ans_0000611-2` (score=0.6997) - In the United States, about 140,000 new cases of colorectal cancer are diagnosed each year. Approximately 3 to 5 percent of these cancers are caused by Lynch syndrome.
4. `medquad_ans_0000037_4-2` (score=0.6977) - Different factors increase or decrease the risk of getting colorectal cancer. Anything that increases your chance of getting a disease is called a risk factor. Anything that decreases your chance of getting a disease is called a protecti...
5. `medquad_ans_0000010-13` (score=0.6800) - Studies show that the following risk factors may increase a person's chances of developing colorectal cancer: age, polyps, personal history, family history, and ulcerative colitis.

**Answer**:

Colorectal cancer is cancer of the colon or rectum, which are parts of the large intestine. It occurs when malignant tumors form in the lining of the large intestine, also called the large bowel [medquad_ans_0000010-10].

**Citations detected**: `medquad_ans_0000010-10`
**Supported citations**: `medquad_ans_0000010-10`
**Unsupported citations**: _none_

## Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.7140) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0002122-5` (score=0.6792) - How might epidermolysis bullosa acquisita be treated?
3. `medquad_ans_0000544-4` (score=0.6686) - Both types of junctional epidermolysis bullosa are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carr...
4. `medquad_ans_0000320-2` (score=0.6572) - The exact prevalence of epidermolysis bullosa simplex is unknown, but this condition is estimated to affect 1 in 30,000 to 50,000 people. The localized type is the most common form of the condition.
5. `medquad_ans_0002121-1` (score=0.6565) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main types of epidermolysis...

**Answer**:

Junctional epidermolysis bullosa is a rare genetic skin disease that causes the skin to blister easily in response to minor injuries or friction. Both types of junctional epidermolysis bullosa are inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs or symptoms [medquad_ans_0000544-4]. This condition affects fewer than 1 per million people in the United States [medquad_ans_0000544-2]. It is one of the four main types of epidermolysis bullosa, along with dystrophic epidermolysis bullosa, epidermolysis bullosa simplex, and Kindler syndrome [medquad_ans_0002121-1].

**Citations detected**: `medquad_ans_0000544-4`, `medquad_ans_0000544-2`, `medquad_ans_0002121-1`
**Supported citations**: `medquad_ans_0000544-4`, `medquad_ans_0000544-2`, `medquad_ans_0002121-1`
**Unsupported citations**: _none_

## Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Top docs**:

1. `medquad_ans_0000668-2` (score=0.6798) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought that the condition is u...
2. `medquad_ans_0000668-4` (score=0.6646) - Molybdenum cofactor deficiency has an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. An affected individual usually inherits one altered copy of the gene from each parent. Par...
3. `medquad_ans_0000668-5` (score=0.6614) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficiency, complementation...
4. `medquad_ans_0000668-3` (score=0.6593) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same signs and symptoms b...
5. `medquad_ans_0000477-3` (score=0.6305) - Hereditary xanthinuria type I is caused by mutations in the XDH gene. This gene provides instructions for making an enzyme called xanthine dehydrogenase. This enzyme is involved in the normal breakdown of purines, which are building bloc...

**Answer**:

Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. These mutations result in reduced or eliminated function of the associated protein, which impairs the biosynthesis of the molybdenum cofactor. The three forms of the disorder—types A, B, and C (complementation groups A, B, and C)—are distinguished by their genetic cause: MOCS1 gene mutations cause type A, MOCS2 gene mutations cause type B, and GPHN gene mutations cause type C [medquad_ans_0000668-3]. Additionally, in some cases, an individual may inherit two mutated copies of the MOCS1 gene through a mechanism called uniparental isodisomy, where the child receives two copies of the mutated gene from one parent due to an error in egg or sperm cell formation [medquad_ans_0

**Citations detected**: `medquad_ans_0000668-3`
**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

## Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.7500) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000755-2` (score=0.7235) - Ornithine transcarbamylase deficiency is believed to occur in approximately 1 in every 80,000 people.
3. `medquad_ans_0000756-3` (score=0.7052) - Mutations in the SLC25A15 gene cause ornithine translocase deficiency. Ornithine translocase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver ce...
4. `medquad_ans_0000755-5` (score=0.6982) - These resources address the diagnosis or management of ornithine transcarbamylase deficiency: - Baby's First Test - Gene Review: Gene Review: Ornithine Transcarbamylase Deficiency - Gene Review: Gene Review: Urea Cycle Disorders Overview...
5. `medquad_ans_0000756-1` (score=0.6964) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is e...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high, and the nervous system is especially sensitive to its effects. This disorder is caused by mutations in the SLC25A15 gene, which results in a defective mitochondrial ornithine transporter. This protein is responsible for moving ornithine across the inner membrane of mitochondria to the mitochondrial matrix, where it participates in the urea cycle. When the transporter is unstable or misshapen, ornithine cannot reach the matrix, disrupting the urea cycle and leading to ammonia buildup. The disorder belongs to a class of genetic diseases called urea cycle disorders and varies widely in severity and age of onset. Infants may present with lethargy, refusal to eat, poor control of breathing or body temperature, seizures,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.7866) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state has its own rules. You...
2. `medquad_ans_0000362-1__dup2` (score=0.7864) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance might be available. Cer...
3. `medquad_ans_0000431-1__dup1` (score=0.7822) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when you need medical care....
4. `medquad_ans_0000869-1__dup1` (score=0.7657) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can bring: - Lists of your c...
5. `medquad_ans_0000196-1__dup3` (score=0.7621) - We all want high-quality health care, but it's hard to know how to choose. There are many things to consider, including - What your insurance covers - Whether a health care provider or service is accredited - The location of a service - ...

**Answer**:

Yes, there is information about financial assistance. Financial assistance may be available if you don't have health insurance or need help with costs that aren't covered. Certain government programs and nonprofit organizations can provide such assistance. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2].

**Citations detected**: `medquad_ans_0000362-1__dup2`
**Supported citations**: `medquad_ans_0000362-1__dup2`
**Unsupported citations**: _none_

## Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Top docs**:

1. `medquad_ans_0004105-2` (score=0.6969) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often accompanied by long-ter...
2. `medquad_ans_0004105-1` (score=0.6801) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About half of affected indivi...
3. `medquad_ans_0000599-1__dup1` (score=0.6705) - Meningococci are a type of bacteria that cause serious infections. The most common infection is meningitis, which is an inflammation of the thin tissue that surrounds the brain and spinal cord. Meningococci can also cause other problems,...
4. `medquad_ans_0000115-3__dup1` (score=0.6436) - The prognosis for for people with encephalitis or meningitis varies. Some cases are mild, short and relatively benign and individuals have full recovery, although the process may be slow. Individuals who experience mild symptoms may reco...
5. `medquad_ans_0000598-1__dup2` (score=0.6280) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enters the body through t...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are the same as those found in other types of meningitis, including high fever, severe headache, nausea, vomiting, stiff neck, photophobia

**Citations detected**: `medquad_ans_0004105-1`
**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

## Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0003319-1` (score=0.6986) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. Unlike other types of...
2. `medquad_ans_0003331-3` (score=0.6820) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...
3. `medquad_ans_0000551-4` (score=0.6796) - When caused by mutations in the ALS2 gene, juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal...
4. `medquad_ans_0000052-1` (score=0.6762) - Amyotrophic lateral sclerosis (ALS) is a progressive disease that affects motor neurons, which are specialized nerve cells that control muscle movement. These nerve cells are found in the spinal cord and the brain. In ALS, motor neurons ...
5. `medquad_ans_0003319-3` (score=0.6734) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality cannot be determine...

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion. Mutations in the ALS2 gene cause juvenile primary lateral sclerosis, which is a related condition [medquad_ans_0003319-1]. However, the specific cause of juvenile ALS is not fully detailed in the provided context. While juvenile primary lateral sclerosis is linked to ALS2 gene mutations, it is not explicitly stated that these mutations are the cause of all forms of juvenile ALS. Therefore, based on the available information, the exact cause of juvenile ALS is not fully specified. [medquad_ans_0003331-3] [medquad_ans_0000551-4]

**Citations detected**: `medquad_ans_0003319-1`, `medquad_ans_0003331-3`, `medquad_ans_0000551-4`
**Supported citations**: `medquad_ans_0003319-1`, `medquad_ans_0003331-3`, `medquad_ans_0000551-4`
**Unsupported citations**: _none_

## Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Top docs**:

1. `medquad_ans_0000809-1__dup2` (score=0.7309) - Summary : Many men have sexual problems. They become more common as men age. Problems can include - Erectile dysfunction - Reduced or lost interest in sex - Problems with ejaculation - Low testosterone Stress, illness, medicines, or emot...
2. `medquad_ans_0000219-11` (score=0.7164) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long enough for sex...
3. `medquad_ans_0000219-4__dup3` (score=0.7145) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men age 60 to 69 - 30 perc...
4. `medquad_ans_0000334-1__dup2` (score=0.7120) - Erectile dysfunction (ED) is a common type of male sexual dysfunction. It is when a man has trouble getting or keeping an erection. ED becomes more common as you get older. But it's not a natural part of aging. Some people have trouble s...
5. `medquad_ans_0000219-5__dup2` (score=0.7078) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain your relationship. Se...

**Answer**:

To know about erectile dysfunction (ED), you should understand that it is a common male sexual dysfunction characterized by difficulty getting or keeping an erection firm enough for sex [medquad_ans_0000219-11]. ED becomes more common with age, affecting about 12% of men younger than 60, 22% of men aged 60 to 69, and 30% of men age 70 or older [medquad_ans_0000219-4__dup3]. It is not a natural part of aging and may indicate underlying health issues such as clogged blood vessels or nerve damage from diabetes [medquad_ans_0000334-1__dup2].

Causes of ED can include stress, illness, medications, emotional problems, and diseases that damage nerves, arteries, or muscles [medquad_ans_00008

**Citations detected**: `medquad_ans_0000219-11`, `medquad_ans_0000219-4__dup3`, `medquad_ans_0000334-1__dup2`
**Supported citations**: `medquad_ans_0000219-11`, `medquad_ans_0000219-4__dup3`, `medquad_ans_0000334-1__dup2`
**Unsupported citations**: _none_

## Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0000306-2` (score=0.6776) - Duchenne and Becker muscular dystrophies together affect 1 in 3,500 to 5,000 newborn males worldwide. Between 400 and 600 boys in the United States are born with these conditions each year.
2. `medquad_ans_0000395-2` (score=0.6760) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dystrophy has an estima...
3. `medquad_ans_0000631-1__dup1` (score=0.6669) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. The different types ...
4. `medquad_ans_0000395-5` (score=0.6616) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscular Dystrophy - Geneti...
5. `medquad_ans_0003601-1` (score=0.6606) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of years. Onset may occu...

**Answer**:

The provided context does not contain any information about the treatments for Ullrich congenital muscular dystrophy. Therefore, a specific answer cannot be given based on the available information. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Top docs**:

1. `medquad_ans_0004005-4` (score=0.6878) - How is microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) inherited? MOPD1 is thought to be inherited in an autosomal recessive manner. This means that affected individuals have abnormal gene changes (mutations) in both cop...
2. `medquad_ans_0004005-3` (score=0.6623) - What causes microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1)? Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) has been shown to be caused by mutations in the RNU4ATAC gene.
3. `medquad_ans_0004005-5` (score=0.6615) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected individuals, with most ...
4. `medquad_ans_0000653-5` (score=0.6553) - These resources address the diagnosis or management of MOPDII: - Genetic Testing Registry: Microcephalic osteodysplastic primordial dwarfism type 2 These resources from MedlinePlus offer information about the diagnosis and management of ...
5. `medquad_ans_0000654-2` (score=0.6411) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.

**Answer**:

The provided context does not contain information about the inheritance pattern of microcephalic osteodysplastic primordial dwarfism type II (MOPDII). Therefore, it is not possible to determine whether MOPDII is inherited based on the given text. [medquad_ans_0000653-5] mentions resources for diagnosis and management of MOPDII but does not specify its inheritance pattern.

Answer: Insufficient context.

**Citations detected**: `medquad_ans_0000653-5`
**Supported citations**: `medquad_ans_0000653-5`
**Unsupported citations**: _none_

## Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000742-2` (score=0.7344) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
2. `medquad_ans_0000742-3` (score=0.7226) - Ochoa syndrome can be caused by mutations in the HPSE2 gene. This gene provides instructions for making a protein called heparanase 2. The function of this protein is not well understood. Mutations in the HPSE2 gene that cause Ochoa synd...
3. `medquad_ans_0000756-2` (score=0.6942) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
4. `medquad_ans_0000746-2` (score=0.6901) - OFCD syndrome is very rare; the incidence is estimated to be less than 1 in 1 million people.
5. `medquad_ans_0004589-4` (score=0.6898) - How is orofaciodigital syndrome type 2 inherited? Orofaciodigital syndrome type 2 is inherited in an autosomal recessive pattern, which means that an individual needs to inherit two mutated (changed) copies of the gene-one from each pare...

**Answer**:

The inheritance pattern of Ochoa syndrome is not specified in the provided context. While the context mentions that some people with Ochoa syndrome do not have mutations in the HPSE2 gene and that the cause in these individuals is unknown, it does not state whether the syndrome is inherited or how it is inherited. Therefore, based on the given information, the inheritance of Ochoa syndrome cannot be determined. [medquad_ans_0000742-3]

**Citations detected**: `medquad_ans_0000742-3`
**Supported citations**: `medquad_ans_0000742-3`
**Unsupported citations**: _none_

## Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Top docs**:

1. `medquad_ans_0003266-1` (score=0.6796) - Isodicentric chromosome 15 syndrome is a chromosome abnormality that affects many different parts of the body. As the name suggests, people with this condition have an extra chromosome (called an isodicentric chromosome 15) which is made...
2. `medquad_ans_0000531-2` (score=0.6778) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
3. `medquad_ans_0000531-3` (score=0.6579) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic material and has two ...
4. `medquad_ans_0000531-4` (score=0.6393) - Isodicentric chromosome 15 syndrome is usually not inherited. The chromosomal change that causes the disorder typically occurs as a random event during the formation of reproductive cells (eggs or sperm) in a parent of the affected indiv...
5. `medquad_ans_0000531-5` (score=0.6269) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management of various health c...

**Answer**:

Treatment is based on the signs and symptoms present in each person [medquad_ans_0003266-1].

**Citations detected**: `medquad_ans_0003266-1`
**Supported citations**: `medquad_ans_0003266-1`
**Unsupported citations**: _none_

## Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Top docs**:

1. `medquad_ans_0006224-4` (score=0.7329) - How is Troyer syndrome inherited? Troyer syndrome is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. Affected people inherit o...
2. `medquad_ans_0000998-2` (score=0.7167) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.
3. `medquad_ans_0000272-2__dup1` (score=0.7144) - There are no specific treatments to prevent or slow the progressive degeneration seen in Troyer syndrome. Symptomatic therapy includes antispasmodic drugs and physical therapy to improve muscle strength and maintain range of motion in th...
4. `medquad_ans_0000272-4__dup1` (score=0.7118) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cause Troyer syndrome a...
5. `medquad_ans_0000964-2` (score=0.7049) - Swyer syndrome occurs in approximately 1 in 80,000 people.

**Answer**:

The cause of Troyer syndrome is mutations in both copies of the responsible gene, inherited in an autosomal recessive manner. Individuals must inherit one mutated copy of the gene from each parent to be affected. Carriers, who have one mutated copy, typically do not show symptoms [medquad_ans_0006224-4]. A gene for Troyer syndrome has been identified, and understanding how this gene causes the disorder is ongoing [medquad_ans_0000272-4__dup1].

**Citations detected**: `medquad_ans_0006224-4`, `medquad_ans_0000272-4__dup1`
**Supported citations**: `medquad_ans_0006224-4`, `medquad_ans_0000272-4__dup1`
**Unsupported citations**: _none_

## Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.6981) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1.2, is one of several...
2. `medquad_ans_0000983-2` (score=0.6793) - Timothy syndrome is a rare condition; fewer than 20 people with this disorder have been reported worldwide. The classic type of Timothy syndrome appears to be more common than the atypical type, which has been identified in only two indi...
3. `medquad_ans_0000983-4` (score=0.6679) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene, and o...
4. `medquad_ans_0003672-1` (score=0.6643) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Timothy syndrome also ...
5. `medquad_ans_0003663-3` (score=0.6627) - What causes long QT syndrome? Acquired long QT syndrome can be caused by certain medicines and medical conditions. Some medications that cause long QT syndrome include antihistamines and decongestants, antibiotics, antidepressants, and c...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal influx of calcium ions into cells. This results in an overload of calcium ions within cardiac muscle cells, altering the heart's rhythm and potentially causing arrhythmia [medquad_ans_0000983-3]. The condition follows an autosomal dominant pattern of inheritance, meaning one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases arise from new mutations and occur in individuals with no family history of the disorder; less commonly, the mutation is inherited from an unaffected parent who is mosaic for the CACNA1C mutation [medquad_ans_0000983-4].

**Citations detected**: `medquad_ans_0000983-3`, `medquad_ans_0000983-4`
**Supported citations**: `medquad_ans_0000983-3`, `medquad_ans_0000983-4`
**Unsupported citations**: _none_

## Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000061-11` (score=0.7180) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your doctor will look for ...
2. `medquad_ans_0000875-2` (score=0.6986) - Rheumatoid arthritis affects about 1.3 million adults in the United States. Worldwide, it is estimated to occur in up to 1 percent of the population. The disease is two to three times more common in women than in men, which may be relate...
3. `medquad_ans_0000061-7__dup1` (score=0.6967) - Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. People with rheumatoid arthritis may feel sick, tired, and sometimes feveris...
4. `medquad_ans_0000061-10` (score=0.6927) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to person, for reasons...
5. `medquad_ans_0000061-1__dup4` (score=0.6896) - An Inflammatory, Autoimmune Disease Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. Rheumatoid arthritis not only affects th...

**Answer**:

Symptoms of rheumatoid arthritis include swelling, warmth, pain, and limitations in joint motion throughout the body. The pain varies from person to person and may be influenced by joint swelling, heat or redness, or joint damage. The disease often causes stiffness and loss of function in the joints. In some cases, it may affect tissues outside the joints, such as the skin, lungs, eyes, and blood vessels. People with rheumatoid arthritis may also feel sick, tired, and sometimes feverish. The condition typically occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is usually affected as well [medquad_ans_0000061-11][medquad_ans_0000061-7__dup1][medquad_ans_0000061-10]medquad_ans_0000061-1\_\_dup

**Citations detected**: `medquad_ans_0000061-11`, `medquad_ans_0000061-7__dup1`, `medquad_ans_0000061-10`
**Supported citations**: `medquad_ans_0000061-11`, `medquad_ans_0000061-7__dup1`, `medquad_ans_0000061-10`
**Unsupported citations**: _none_

## Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Top docs**:

1. `medquad_ans_0000149-1__dup3` (score=0.6671) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, digestive tract, and ...
2. `medquad_ans_0002796-4` (score=0.6664) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti-inflammatory drugs ...
3. `medquad_ans_0002796-1` (score=0.6630) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, ...
4. `medquad_ans_0002796-3` (score=0.6488) - Can Henoch-Schonlein purpura be inherited? The cause of Henoch-Schonlein purpura is currently unknown. Some evidence suggests that genetic predisposition may contribute to the development of this disease in some cases. Only a few familie...
5. `medquad_ans_0000152-4__dup2` (score=0.6456) - IgA nephropathy is one of the most common kidney diseases, other than those caused by diabetes or high blood pressure.1 IgA nephropathy can occur at any age, although the first evidence of kidney disease most frequently appears when peop...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, digestive tract, and joints. It is most common in children between 2 and 6 years of age, though it can occur at any age. Most people recover completely from HSP, though kidney damage is the most likely long-term complication. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage renal disease when treated with dialysis or a kidney transplant. The cause is not fully understood but may develop as an immune response to an infection. Some evidence suggests a possible genetic predisposition, though no specific genes have been proven to increase the risk. There is no cure for HSP; treatment focuses on relieving symptoms such as pain

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Top docs**:

1. `medquad_ans_0003162-1` (score=0.7036) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunodeficiency cant get ri...
2. `medquad_ans_0001052-5` (score=0.6854) - These resources address the diagnosis or management of X-linked hyper IgM syndrome: - Gene Review: Gene Review: X-Linked Hyper IgM Syndrome - Genetic Testing Registry: Immunodeficiency with hyper IgM type 1 - MedlinePlus Encyclopedia: Im...
3. `medquad_ans_0001052-2` (score=0.6623) - X-linked hyper IgM syndrome is estimated to occur in 2 per million newborn boys.
4. `medquad_ans_0003162-3` (score=0.6595) - What causes hyper IgM syndrome? A flawed gene (or genes) in T-cells (a type of white blood cell that is part of the immune system and helps the body fight diseases or harmful substances) is responsible for hyper IgM syndrome. The faulty ...
5. `medquad_ans_0003163-1` (score=0.6572) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information is available, the ta...

**Answer**:

The provided context does not contain information about the symptoms of Immunodeficiency with hyper IgM type 5. [medquad_ans_0003162-1] mentions the general characteristics of hyper IgM syndrome, including elevated immunoglobulin M and absence of IgG, A, and E, and susceptibility to bacterial and opportunistic infections, but does not specify symptoms for type 5. [medquad_ans_0003162-3] discusses X-linked hyper IgM syndrome and general inheritance patterns, and [medquad_ans_0001052-5] lists resources for diagnosis and management but does not detail symptoms for type 5. Therefore, based on the available context, the symptoms of Immunodeficiency with hyper IgM type 5 cannot be determined.

**Citations detected**: `medquad_ans_0003162-1`, `medquad_ans_0003162-3`, `medquad_ans_0001052-5`
**Supported citations**: `medquad_ans_0003162-1`, `medquad_ans_0003162-3`, `medquad_ans_0001052-5`
**Unsupported citations**: _none_

## Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Top docs**:

1. `medquad_ans_0000017-3` (score=0.6934) - In very few families, people develop AD in their 30s, 40s, and 50s. This is known as "early onset" AD. These individuals have a mutation in one of three different inherited genes that causes the disease to begin at an earlier age. More t...
2. `medquad_ans_0000896-5` (score=0.6869) - These resources address the diagnosis or management of Senior-Lken syndrome: - Genetic Testing Registry: Senior-Loken syndrome 1 - Genetic Testing Registry: Senior-Loken syndrome 3 - Genetic Testing Registry: Senior-Loken syndrome 4 - Ge...
3. `medquad_ans_0000896-2` (score=0.6816) - Senior-Lken syndrome is a rare disorder, with an estimated prevalence of about 1 in 1 million people worldwide. Only a few families with the condition have been described in the medical literature.
4. `medquad_ans_0000563-4` (score=0.6791) - Klippel-Trenaunay syndrome is almost always sporadic, which means that it occurs in people with no history of the disorder in their family. Studies suggest that the condition may result from gene mutations that are not inherited. These g...
5. `medquad_ans_0003474-3` (score=0.6785) - How is Larson syndrome inherited? Larson syndrome is inherited in an autosomal dominant manner. A condition is autosomal dominant when having one copy of the changed (mutated) gene in each cell is enough to cause signs or symptoms of the...

**Answer**:

The provided context does not contain information about the genetic changes related to Senior-Lken syndrome. [medquad_ans_0000896-2] mentions that Senior-Lken syndrome is a rare disorder with an estimated prevalence of about 1 in 1 million people worldwide and that only a few families have been described, but it does not specify any genetic changes. Therefore, based on the available context, the genetic changes related to Senior-Lken syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000896-2`
**Supported citations**: `medquad_ans_0000896-2`
**Unsupported citations**: _none_

## Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.7485) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed as having lymphedema ...
2. `medquad_ans_0002834-3` (score=0.7212) - What causes hereditary lymphedema type II? The cause of hereditary lymphedema type II is unknown. The condition is thought to be genetic because it tends to run in families. Researchers have studied many genes associated with the lymphat...
3. `medquad_ans_0000610-5` (score=0.7164) - These resources address the diagnosis or management of lymphedema-distichiasis syndrome: - Gene Review: Gene Review: Lymphedema-Distichiasis Syndrome - Genetic Testing Registry: Distichiasis-lymphedema syndrome - MedlinePlus Encyclopedia...
4. `medquad_ans_0000660-3` (score=0.7101) - Mutations in the FLT4 gene cause some cases of Milroy disease. The FLT4 gene provides instructions for producing a protein called vascular endothelial growth factor receptor 3 (VEGFR-3), which regulates the development and maintenance of...
5. `medquad_ans_0000610-1` (score=0.7083) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and immune cells through...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown. [medquad_ans_0000610-2]

**Citations detected**: `medquad_ans_0000610-2`
**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

## Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Top docs**:

1. `medquad_ans_0000397-1__dup1` (score=0.7503) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourself or a family membe...
2. `medquad_ans_0000013-25` (score=0.7473) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for treatment or monitor...
3. `medquad_ans_0000399-1__dup1` (score=0.7430) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn babies - Finding out ...
4. `medquad_ans_0001717-5` (score=0.7428) - Is genetic testing available for Danon disease? Yes. GeneTests lists laboratories offering clinical genetic testing for Danon disease. Clinical genetic tests are ordered to help diagnose a person or family and to aid in decisions regardi...
5. `medquad_ans_0000049-4__dup4` (score=0.7252) - Testing for monogenic diabetes involves providing a blood sample from which DNA is isolated. The DNA is analyzed for changes in the genes that cause monogenic diabetes. Abnormal results can determine the gene responsible for diabetes in ...

**Answer**:

Yes, there is information about genetic testing. Genetic testing involves tests on blood and other tissue to find genetic disorders. Over 2000 tests are available, and doctors use them for several reasons, including finding genetic diseases in unborn babies, determining if people carry a gene for a disease that might be passed on to their children, screening embryos for disease, testing for genetic diseases in adults before symptoms appear, making a diagnosis in someone who has disease symptoms, and determining the type or dose of medicine that is best for a certain person [medquad_ans_0000399-1__dup1]. Genetic testing may also serve as a relief by eliminating uncertainty about a health condition, help doctors make treatment or monitoring recommendations, provide information for making health and family decisions, assist in taking steps to lower the chance of developing a disease through earlier screening or lifestyle changes, and support informed choices about the future, such as whether to have a baby

**Citations detected**: `medquad_ans_0000399-1__dup1`
**Supported citations**: `medquad_ans_0000399-1__dup1`
**Unsupported citations**: _none_

## Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Top docs**:

1. `medquad_ans_0000654-2` (score=0.6762) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
2. `medquad_ans_0000149-1` (score=0.6661) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxygen-rich blood from ...
3. `medquad_ans_0000654-1` (score=0.6526) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary malformations). In peo...
4. `medquad_ans_0000654-5` (score=0.6488) - These resources address the diagnosis or management of microcephaly-capillary malformation syndrome: - Gene Review: Gene Review: Microcephaly-Capillary Malformation Syndrome - Genetic Testing Registry: Microcephaly-capillary malformation...
5. `medquad_ans_0000149-5` (score=0.6453) - These resources address the diagnosis or management of CM-AVM: - Gene Review: Gene Review: RASA1-Related Disorders - Genetic Testing Registry: Capillary malformation-arteriovenous malformation These resources from MedlinePlus offer infor...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or develop during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries affecting blood circulation. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications of

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Top docs**:

1. `medquad_ans_0000309-2` (score=0.7208) - The prevalence of dystonia 6 is unknown. Studies indicate that it likely accounts for between 1 and 3 percent of all cases of dystonia. For reasons that are unclear, the disorder appears to be slightly more prevalent in females than in m...
2. `medquad_ans_0000299-2` (score=0.7173) - Dopa-responsive dystonia is estimated to affect 1 per million people worldwide. However, the disorder is likely underdiagnosed because the condition may not be identified in people with mild symptoms, or it may be misdiagnosed in people ...
3. `medquad_ans_0000704-2` (score=0.7141) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
4. `medquad_ans_0000312-2` (score=0.7081) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,000 people in this p...
5. `medquad_ans_0000859-2` (score=0.6993) - Rapid-onset dystonia parkinsonism appears to be a rare disorder, although its prevalence is unknown. It has been diagnosed in individuals and families from the United States, Europe, and Korea.

**Answer**:

The provided context does not contain information about the symptoms of Dystonia 16. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.7231) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable. Vision, phy...
2. `medquad_ans_0000001-3__dup1` (score=0.6981) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-threatening.
3. `medquad_ans_0000899-2` (score=0.6895) - Septo-optic dysplasia has a reported incidence of 1 in 10,000 newborns.
4. `medquad_ans_0000001-2__dup1` (score=0.6889) - Absence of the SP alone is not a disorder but is instead a characteristic noted in children with septo-optic dysplasia or other developmental anomalies.
5. `medquad_ans_0005548-4` (score=0.6856) - Is septo-optic dysplasia inherited?

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Citations detected**: `medquad_ans_0005548-5`
**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

## Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Top docs**:

1. `medquad_ans_0000208-4__dup2` (score=0.6604) - Benign prostatic hyperplasia is the most common prostate problem for men older than age 50. In 2010, as many as 14 million men in the United States had lower urinary tract symptoms suggestive of benign prostatic hyperplasia.1 Although be...
2. `medquad_ans_0000208-12` (score=0.6444) - Researchers have not found a way to prevent benign prostatic hyperplasia. Men with risk factors for benign prostatic hyperplasia should talk with a health care provider about any lower urinary tract symptoms and the need for regular pros...
3. `medquad_ans_0000208-1__dup4` (score=0.6356) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic obstruction. The pro...
4. `medquad_ans_0000208-7` (score=0.6299) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stone...
5. `medquad_ans_0000058-19` (score=0.6294) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find out if you need tre...

**Answer**:

The complications of benign prostatic hyperplasia may include - acute urinary retention, chronic, or long-lasting, urinary retention, blood in the urine, urinary tract infections (UTIs), bladder damage, kidney damage, and bladder stones. Most men with benign prostatic hyperplasia do not develop these complications. However, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

**Citations detected**: `medquad_ans_0000208-7`
**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

## Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Top docs**:

1. `medquad_ans_0000001-3__dup1` (score=0.6711) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-threatening.
2. `medquad_ans_0000001-1__dup1` (score=0.6057) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers that connect the cere...
3. `medquad_ans_0000899-1` (score=0.6050) - Septo-optic dysplasia is a disorder of early brain development. Although its signs and symptoms vary, this condition is traditionally defined by three characteristic features: underdevelopment (hypoplasia) of the optic nerves, abnormal f...
4. `medquad_ans_0000150-2__dup1` (score=0.6007) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
5. `medquad_ans_0000001-2__dup1` (score=0.5978) - Absence of the SP alone is not a disorder but is instead a characteristic noted in children with septo-optic dysplasia or other developmental anomalies.

**Answer**:

The provided context does not mention any treatments for the absence of the septum pellucidum. Therefore, based on the available information, it is not possible to determine the treatments for this condition. [medquad_ans_0000001-3\_\_dup1][medquad_ans_0000001-1__dup1][medquad_ans_000899-1]

**Citations detected**: `medquad_ans_0000001-3__dup1`, `medquad_ans_0000001-1__dup1`, `medquad_ans_000899-1`
**Supported citations**: `medquad_ans_0000001-3__dup1`, `medquad_ans_0000001-1__dup1`
**Unsupported citations**: `medquad_ans_000899-1`

## Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Top docs**:

1. `medquad_ans_0002322-1` (score=0.7438) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sores, then form a red-...
2. `medquad_ans_0002322-5` (score=0.6973) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the benefit of the therapie...
3. `medquad_ans_0002322-3` (score=0.6875) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD and Epstein-Barr vir...
4. `medquad_ans_0004907-2` (score=0.6762) - What are the signs and symptoms of pityriasis lichenoides et varioliformis acuta? Pityriasis lichenoides et varioliformis acuta (PLEVA) is the acute form of a skin condition called pityriasis lichenoides. It is characterized by the sudde...
5. `medquad_ans_0002322-4` (score=0.6747) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FUMHD are outlined bel...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include fever, extensive and painful loss of skin tissue, ulceration, breakdown of skin forming open sores, and the formation of red-brown crusts. The disease often begins as pityriasis lichenoides et varioliformis acuta (PLEVA) and then rapidly progresses to large, destructive ulcers. There may also be secondary infection of the ulcers [medquad_ans_0002322-1].

**Citations detected**: `medquad_ans_0002322-1`
**Supported citations**: `medquad_ans_0002322-1`
**Unsupported citations**: _none_

## Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Top docs**:

1. `medquad_ans_0002833-4` (score=0.6677) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is...
2. `medquad_ans_0002833-1` (score=0.6572) - Hereditary leiomyomatosis and renal cell cancer (HLRCC) is a condition that causes benign tumors of smooth muscle tissue in the skin (cutaneous leiomyomas) and in the uterus in females (uterine leiomyomas, or fibroids). The condition als...
3. `medquad_ans_0000465-5` (score=0.6460) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - MedlinePlus Encyclope...
4. `medquad_ans_0000465-1` (score=0.6209) - Hereditary leiomyomatosis and renal cell cancer (HLRCC) is a disorder in which affected individuals tend to develop benign tumors containing smooth muscle tissue (leiomyomas) in the skin and, in females, the uterus. This condition also i...
5. `medquad_ans_0002833-2` (score=0.6190) - What are the signs and symptoms of Hereditary leiomyomatosis and renal cell cancer? Signs and symptoms of hereditary leiomyomatosis and renal cell cancer (HLRCC) typically begin in adulthood at an average age of 25. The skin growths (cut...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4]. This inheritance pattern results in each child of an affected individual having a 50% (1 in 2) risk to inherit the mutated gene [medquad_ans_0002833-4]. HLRCC is caused by mutations in the FH gene and is inherited in an autosomal dominant manner [medquad_ans_0002833-1].

**Citations detected**: `medquad_ans_0002833-4`, `medquad_ans_0002833-1`
**Supported citations**: `medquad_ans_0002833-4`, `medquad_ans_0002833-1`
**Unsupported citations**: _none_

## Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Top docs**:

1. `medquad_ans_0000389-2` (score=0.7101) - Frontometaphyseal dysplasia is a rare disorder; only a few dozen cases have been reported worldwide.
2. `medquad_ans_0000389-5` (score=0.6725) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resources from MedlinePlus ...
3. `medquad_ans_0000390-2` (score=0.6723) - Frontonasal dysplasia is likely a rare condition; at least 100 cases have been reported in the scientific literature.
4. `medquad_ans_0002374-1` (score=0.6665) - Fibrous dysplasia is a skeletal disorder that is characterized by the replacement of normal bone with fibrous bone tissue. It may involve one bone (monostotic) or multiple bones (polyostotic). Fibrous dysplasia can affect any bone in the...
5. `medquad_ans_0000250-2` (score=0.6587) - Craniometaphyseal dysplasia is a very rare disorder; its incidence is unknown.

**Answer**:

The provided context does not specify the symptoms of frontometaphyseal dysplasia. [medquad_ans_0000389-2] states it is a rare disorder with few reported cases, and [medquad_ans_0000389-5] lists resources for diagnosis and management, but no details on symptoms are given. Therefore, based on the available information, the symptoms of frontometaphyseal dysplasia cannot be determined.

**Citations detected**: `medquad_ans_0000389-2`, `medquad_ans_0000389-5`
**Supported citations**: `medquad_ans_0000389-2`, `medquad_ans_0000389-5`
**Unsupported citations**: _none_

## Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Top docs**:

1. `medquad_ans_0000128-5__dup1` (score=0.7100) - Many people get diverticular disease. Starting at age 40, the chance of getting it increases about every 10 years. About half of people between the ages of 60 and 80 have diverticular disease. Almost everyone over 80 has it.
2. `medquad_ans_0000128-7` (score=0.7022) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding, you need to go to t...
3. `medquad_ans_0000097-5__dup1` (score=0.6909) - Diverticulosis becomes more common as people age, particularly in people older than age 50.3 Some people with diverticulosis develop diverticulitis, and the number of cases is increasing. Although diverticular disease is generally though...
4. `medquad_ans_0000128-11` (score=0.6886) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain foods, high-fiber fru...
5. `medquad_ans_0000128-2__dup2` (score=0.6804) - Doctors are not sure what causes diverticular disease. Many think a diet low in fiber is the main cause. Fiber is a part of food that your body cannot digest. It is found in many fruits and vegetables. Fiber stays in the colon and absorb...

**Answer**:

Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet, particularly by adopting a high-fiber diet and using pain medication. Add whole grain foods, high-fiber fruits, and vegetables to your diet. If you have rectal bleeding, you need to go to the hospital so a doctor can find the part of your colon that is bleeding. The doctor may use a special drug that makes the bleeding stop or may decide to operate and remove the part of the colon that is bleeding. Contact a doctor if you notice symptoms such as fever, chills, nausea, vomiting, abdominal pain, rectal bleeding, or a change in bowel habits [medquad_ans_0000128-7, medquad_ans_0000128-11].

**Citations detected**: `medquad_ans_0000128-7, medquad_ans_0000128-11`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000128-7, medquad_ans_0000128-11`

## Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Top docs**:

1. `medquad_ans_0000730-2` (score=0.6871) - Nicolaides-Baraitser syndrome is likely a rare condition; approximately 75 cases have been reported in the scientific literature.
2. `medquad_ans_0000107-2` (score=0.6826) - Baraitser-Winter syndrome is a rare condition. Fewer than 50 cases have been reported in the medical literature.
3. `medquad_ans_0000127-2` (score=0.6754) - Bjrnstad syndrome is a rare condition, although its prevalence is unknown. It has been found in populations worldwide.
4. `medquad_ans_0000107-1` (score=0.6678) - Baraitser-Winter syndrome is a condition that affects the development of many parts of the body, particularly the face and the brain. An unusual facial appearance is the most common characteristic of Baraitser-Winter syndrome. Distinctiv...
5. `medquad_ans_0005571-3` (score=0.6675) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of the disease, and other...

**Answer**:

The provided context does not mention "Baraitser Brett Piesowicz syndrome" or any symptoms associated with this condition. Therefore, based on the given information, it is not possible to answer the question. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Top docs**:

1. `medquad_ans_0003911-2` (score=0.6648) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a meningioma. When treatm...
2. `medquad_ans_0003911-1` (score=0.6021) - Meningiomas originate in the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign, though a minority of meningiomas can be classified as atypical or malignant. Though rare, malignant meningiomas ca...
3. `medquad_ans_0000052-2__dup1` (score=0.5954) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
4. `medquad_ans_0000035_1-7` (score=0.5896) - Key Points - There are different types of treatment for patients with pituitary tumors. - Four types of standard treatment are used: - Surgery - Radiation therapy - Drug therapy - Chemotherapy - New types of treatment are being tested in...
5. `medquad_ans_0000724-2` (score=0.5861) - Neurofibromatosis type 1 occurs in 1 in 3,000 to 4,000 people worldwide.

**Answer**:

The treatments for meningioma vary depending on the location of the meningioma and the symptoms it causes. Careful observation may be the best course of action for some patients. When treatment is necessary, surgery and radiation are the most common forms of treatment. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. It is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2]. Steroids may also be prescribed to reduce tumor-related swelling in the central nervous system [medquad_ans_0000

**Citations detected**: `medquad_ans_0003911-2`
**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

## Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.7125) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a team of health care pr...
2. `medquad_ans_0000006_6-8` (score=0.6896) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health care providers who ...
3. `medquad_ans_0000006_7-2` (score=0.6834) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Staging ...
4. `medquad_ans_0000006_7-3` (score=0.6820) - The cause of most childhood CNS germ cell tumors is not known.
5. `medquad_ans_0000015_1-8` (score=0.6645) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers who are experts in ...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials for new treatments [medquad_ans_0000006_7-8]. Treatment is typically planned by a team of health care providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8]. Some treatments are standard, while others are being tested in clinical trials [medquad_ans_0000006_7-8]. Patients may consider participating in clinical trials before, during, or after starting treatment [

**Citations detected**: `medquad_ans_0000006_7-8`
**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

## Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Top docs**:

1. `medquad_ans_0000729-2` (score=0.7482) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
2. `medquad_ans_0004401-1` (score=0.7352) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condit...
3. `medquad_ans_0004401-4` (score=0.7281) - How is neutral lipid storage disease with myopathy inherited? This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal ...
4. `medquad_ans_0004401-3` (score=0.7233) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose triglyceride lipase (ATGL...
5. `medquad_ans_0000729-3` (score=0.7158) - Mutations in the PNPLA2 gene cause neutral lipid storage disease with myopathy. The PNPLA2 gene provides instructions for making an enzyme called adipose triglyceride lipase (ATGL). The ATGL enzyme plays a role in breaking down fats call...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene, which provides instructions for making an enzyme called adipose triglyceride lipase (ATGL). The ATGL enzyme plays a role in breaking down fats called triglycerides, which are an important source of stored energy in cells. Mutations in the PNPLA2 gene impair the ATGL enzyme's ability to break down triglycerides, leading to their accumulation in muscle and tissues, resulting in the signs and symptoms of the condition. It is inherited in an autosomal recessive pattern [medquad_ans_0004401-1, medquad_ans_0004401-3, medquad_ans_00

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.6677) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. Cases of Zika virus ...
2. `medquad_ans_0006546-2` (score=0.6385) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, drink fluids, and take...
3. `medquad_ans_0000981-1__dup1` (score=0.6274) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus h...
4. `medquad_ans_0000174-1__dup3` (score=0.5961) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected blood. There have b...
5. `medquad_ans_0000090-2__dup2` (score=0.5854) - Despite a vigorous search, scientists have not yet identified what causes CFS. While a single cause for CFS may yet be identified, another possibility is that CFS has multiple causes. Conditions that have been studied to determine if the...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and via blood transfusions. The illness associated with Zika virus infection is usually mild, with common symptoms including fever, rash, joint pain, and conjunctivitis (red eyes). Recent research has suggested a link between Zika virus infection and Guillain-Barre syndrome (GBS), and prenatal infection has been shown to cause adverse pregnancy outcomes such as microcephaly and other serious brain anomalies. There is no vaccine or specific medicine to treat Zika virus; management includes rest, hydration, and use of acetaminophen for pain, with avoidance of aspirin and NSAIDs until dengue is ruled out. [medquad_ans_0006546-1]medquad_ans_0006

**Citations detected**: `medquad_ans_0006546-1`
**Supported citations**: `medquad_ans_0006546-1`
**Unsupported citations**: _none_

## Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Top docs**:

1. `medquad_ans_0000019-4__dup2` (score=0.7377) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not apply to broken heart sy...
2. `medquad_ans_0000019-3__dup2` (score=0.7006) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack, what is happening ...
3. `medquad_ans_0000019-6` (score=0.6986) - Because the symptoms are similar, at first your doctor may not be able to tell whether you are experiencing broken heart syndrome or having a heart attack. Therefore, the doctors immediate goals will be: To determine whats causing your s...
4. `medquad_ans_0000019-7` (score=0.6920) - Even though broken heart syndrome may feel like a heart attack, its a very different problem that needs a different type of treatment. The good news is that broken heart syndrome is usually treatable, and most people make a full recovery...
5. `medquad_ans_0000443-1__dup1` (score=0.6854) - In the United States, 1 in 4 women dies from heart disease. The most common cause of heart disease in both men and women is narrowing or blockage of the coronary arteries, the blood vessels that supply blood to the heart itself. This is ...

**Answer**:

The cause of broken heart syndrome is not fully known, but extreme emotional or physical stress is believed to play a role. Sudden stress is thought to release hormones such as adrenaline and noradrenaline, which can overwhelm or stun the heart, leading to temporary changes in heart muscle cells or coronary blood vessels. This may result in the heart becoming weak, with the left ventricle bulging and unable to pump blood effectively. Potential causes under investigation include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Citations detected**: `medquad_ans_0000019-3__dup2`
**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

## Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Top docs**:

1. `medquad_ans_0000448-2` (score=0.7758) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid underactivity (hypothyroi...
2. `medquad_ans_0002735-3` (score=0.7689) - What causes Hashimoto's encephalitis? The exact cause of Hashimoto's encephalitis (HE) is unknown, but is thought to relate to autoimmune or other autoinflammatory processes. While it is associated with Hashimoto's thyroiditis, the exact...
3. `medquad_ans_0002735-1` (score=0.7570) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and paranoid delusions, has al...
4. `medquad_ans_0000005-9__dup1` (score=0.7507) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is a disorder that occ...
5. `medquad_ans_0002736-4` (score=0.7460) - Is Hashimoto's syndrome inherited? The inheritance pattern of Hashimoto's syndrome is unclear because many genetic and environmental factors appear to be involved. However, the condition can cluster in families, and having a close relati...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease characterized by chronic lymphocytic thyroiditis. It affects 1 to 2 percent of people in the United States and is more common in women than in men, which may be related to hormonal factors. It is the most common cause of hypothyroidism in the United States, occurring when the thyroid does not produce enough thyroid hormone for the body's needs. Many individuals with Hashimoto thyroiditis have no symptoms initially, and as the disease progresses, the thyroid may enlarge, leading to a goiter that can cause a feeling of fullness in the throat. Not everyone with Hashimoto thyroiditis develops hypothyroidism, and when it does, the condition may be subclinical and mild, especially early in its course. [medquad_ans_0000448-2][medquad_ans_0000005-9__dup1]

**Citations detected**: `medquad_ans_0000448-2`, `medquad_ans_0000005-9__dup1`
**Supported citations**: `medquad_ans_0000448-2`, `medquad_ans_0000005-9__dup1`
**Unsupported citations**: _none_

## Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Top docs**:

1. `medquad_ans_0001043-2` (score=0.7581) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
2. `medquad_ans_0000002-3__dup1` (score=0.7570) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
3. `medquad_ans_0001043-3` (score=0.7227) - Mutations in the LIPA gene cause Wolman disease. The LIPA gene provides instructions for producing an enzyme called lysosomal acid lipase. This enzyme is found in the lysosomes (compartments that digest and recycle materials in the cell)...
4. `medquad_ans_0001043-5` (score=0.7215) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management of various health c...
5. `medquad_ans_0000002-2__dup1` (score=0.7211) - Enzyme replacement therapy for both Wolman's and cholesteryl ester storage disease is currently under investigation. Certain drugs may be given to help with adrenal gland production, and children may need to be fed intravenously. Individ...

**Answer**:

Wolman disease is a rare genetic disorder caused by mutations in the LIPA gene, which leads to a deficiency of lysosomal acid lipase. This enzyme is essential for breaking down lipids such as cholesteryl esters and triglycerides in lysosomes, the cellular compartments responsible for digestion and recycling. A shortage of this enzyme results in the accumulation of fats within cells and tissues, leading to malnutrition and the development of signs and symptoms of the disease. It is estimated to occur in 1 in 350,000 newborns and is usually fatal by age 1 [medquad_ans_0001043-2, medquad_ans_0001043-3]. Management and diagnosis may involve genetic testing, enzyme replacement therapy (under investigation), intravenous feeding, and supportive care such as genetic counseling and palliative care [medquad_ans_0001043

**Citations detected**: `medquad_ans_0001043-2, medquad_ans_0001043-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0001043-2, medquad_ans_0001043-3`

## Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Top docs**:

1. `medquad_ans_0002753-1` (score=0.6476) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors are typically slow-...
2. `medquad_ans_0002753-3` (score=0.6351) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational viny...
3. `medquad_ans_0002753-4` (score=0.6299) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possible in only 50-67% o...
4. `medquad_ans_0000181-2` (score=0.6003) - Chordomas are rare, occurring in approximately 1 per million individuals each year. Chordomas comprise fewer than 1 percent of tumors affecting the brain and spinal cord.
5. `medquad_ans_0000060-3__dup1` (score=0.5967) - Rebleeding from a cavernous angioma is common, it is not predictable, and individuals frequently have multiple CCMs found via magnetic resonance imaging. Individuals with CCM are faced with a diagnosis that imparts risk of multiple futur...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Citations detected**: `medquad_ans_0002753-3`
**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

## Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Top docs**:

1. `medquad_ans_0000655-3` (score=0.6934) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or more genes. Most ge...
2. `medquad_ans_0000655-4` (score=0.6727) - Isolated microphthalmia is sometimes inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of ...
3. `medquad_ans_0000655-2` (score=0.6608) - Microphthalmia occurs in approximately 1 in 10,000 individuals.
4. `medquad_ans_0000593-2` (score=0.6591) - Lenz microphthalmia syndrome is a very rare condition; its incidence is unknown. It has been identified in only a few families worldwide.
5. `medquad_ans_0000248-4` (score=0.6581) - Craniofacial microsomia most often occurs in a single individual in a family and is not inherited. If the condition is caused by a chromosomal abnormality, it may be inherited from one affected parent or it may result from a new abnormal...

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. These changes may occur in an autosomal recessive pattern, where both copies of the gene in each cell have mutations, and the parents of affected individuals typically carry one copy of the mutated gene without showing signs or symptoms. Microphthalmia may also result from chromosomal abnormalities affecting one or more genes. However, most genetic changes associated with isolated microphthalmia have been identified in only very small numbers of affected individuals [medquad_ans_0000655-3][medquad_ans_0000655-4].

**Citations detected**: `medquad_ans_0000655-3`, `medquad_ans_0000655-4`
**Supported citations**: `medquad_ans_0000655-3`, `medquad_ans_0000655-4`
**Unsupported citations**: _none_

## Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Top docs**:

1. `medquad_ans_0000028-6__dup1` (score=0.7240) - Because uric acids role in gout is well understood and medications to ease attacks and reduce the risk or severity of future attacks are widely available, gout is one of the mostif not the mostcontrollable forms of arthritis. But researc...
2. `medquad_ans_0000028-8` (score=0.7104) - Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by stressful events, alcohol or drugs, or the presenc...
3. `medquad_ans_0000028-9` (score=0.6957) - Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines, which are part of all human tissue and are found in many foods, especial...
4. `medquad_ans_0000028-14` (score=0.6945) - To confirm a diagnosis of gout, the doctor inserts a needle into the inflamed joint and draws a sample of synovial fluid, the substance that lubricates a joint. A laboratory technician places some of the fluid on a slide and looks for ur...
5. `medquad_ans_0000028-2__dup3` (score=0.6875) - A Buildup of Uric Acid Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines which are a part of all human tissue and found in ...

**Answer**:

The provided context does not mention the symptoms of gout. [medquad_ans_0000028-6__dup1]

**Citations detected**: `medquad_ans_0000028-6__dup1`
**Supported citations**: `medquad_ans_0000028-6__dup1`
**Unsupported citations**: _none_

## Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Top docs**:

1. `medquad_ans_0000309-2` (score=0.7174) - The prevalence of dystonia 6 is unknown. Studies indicate that it likely accounts for between 1 and 3 percent of all cases of dystonia. For reasons that are unclear, the disorder appears to be slightly more prevalent in females than in m...
2. `medquad_ans_0000312-2` (score=0.7002) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,000 people in this p...
3. `medquad_ans_0000299-2` (score=0.6959) - Dopa-responsive dystonia is estimated to affect 1 per million people worldwide. However, the disorder is likely underdiagnosed because the condition may not be identified in people with mild symptoms, or it may be misdiagnosed in people ...
4. `medquad_ans_0000704-2` (score=0.6918) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
5. `medquad_ans_0000859-2` (score=0.6849) - Rapid-onset dystonia parkinsonism appears to be a rare disorder, although its prevalence is unknown. It has been diagnosed in individuals and families from the United States, Europe, and Korea.

**Answer**:

The context provided does not contain information about the causes of Dystonia 8. Therefore, the cause cannot be determined from the given text. [doc_id: not available]

**Citations detected**: `doc_id: not available`
**Supported citations**: _none_
**Unsupported citations**: `doc_id: not available`

## Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.7273) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unknown.
2. `medquad_ans_0000612-5` (score=0.7100) - These resources address the diagnosis or management of lysinuric protein intolerance: - Gene Review: Gene Review: Lysinuric Protein Intolerance - Genetic Testing Registry: Lysinuric protein intolerance - MedlinePlus Encyclopedia: Aminoac...
3. `medquad_ans_0000612-1` (score=0.6847) - Lysinuric protein intolerance is a disorder caused by the body's inability to digest and use certain protein building blocks (amino acids), namely lysine, arginine, and ornithine. Because the body cannot effectively break down these amin...
4. `medquad_ans_0000495-1` (score=0.6840) - Hyperlysinemia is an inherited condition characterized by elevated blood levels of the amino acid lysine, a building block of most proteins. Hyperlysinemia is caused by the shortage (deficiency) of the enzyme that breaks down lysine. Hyp...
5. `medquad_ans_0003719-1` (score=0.6831) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids, which are found in ...

**Answer**:

Lysinuric protein intolerance affects approximately 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Citations detected**: `medquad_ans_0000612-2`
**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

## Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Top docs**:

1. `medquad_ans_0000170-10` (score=0.6475) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart to arteries througho...
2. `medquad_ans_0000170-1__dup3` (score=0.6172) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries ...
3. `medquad_ans_0005299-4` (score=0.6098) - How is Renal nutcracker syndrome diagnosed? A diagnosis of renal nutcracker syndrome is often suspected based on the presence of characteristic signs and symptoms once other conditions that cause similar features have been ruled out. Add...
4. `medquad_ans_0000170-6` (score=0.5890) - People with RAS are at increased risk for complications resulting from loss of kidney function or atherosclerosis occurring in other blood vessels, such as - chronic kidney disease (CKD)reduced kidney function over a period of time - cor...
5. `medquad_ans_0000123-3__dup1` (score=0.5880) - Individuals with Fabry disease often die prematurely of complications from strokes, heart disease, or kidney failure.

**Answer**:

The first symptoms of Renal Artery Stenosis (RAS) are usually either high blood pressure or decreased kidney function, or both [medquad_ans_0000170-10]. RAS often has no symptoms until it becomes severe [medquad_ans_0000170-10].

**Citations detected**: `medquad_ans_0000170-10`
**Supported citations**: `medquad_ans_0000170-10`
**Unsupported citations**: _none_

## Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Top docs**:

1. `medquad_ans_0000691-2` (score=0.6900) - Multiple lentigines syndrome is thought to be a rare condition; approximately 200 cases have been reported worldwide.
2. `medquad_ans_0000712-5` (score=0.6897) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis - Genetic Testing Reg...
3. `medquad_ans_0001011-2` (score=0.6819) - UV-sensitive syndrome appears to be a rare condition; only a small number of affected individuals have been reported in the scientific literature. However, this condition may be underdiagnosed.
4. `medquad_ans_0000009-2` (score=0.6811) - Acral peeling skin syndrome is a rare condition, with several dozen cases reported in the medical literature. However, because its signs and symptoms tend to be mild and similar to those of other skin disorders, the condition is likely u...
5. `medquad_ans_0001066-2` (score=0.6803) - Xeroderma pigmentosum is a rare disorder; it is estimated to affect about 1 in 1 million people in the United States and Europe. The condition is more common in Japan, North Africa, and the Middle East.

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that the Genetic Testing Registry includes information on these conditions, no details about diagnosis, management, or treatment are given [medquad_ans_0000712-5]. Therefore, based on the available information, the treatments for these syndromes cannot be determined.

**Citations detected**: `medquad_ans_0000712-5`
**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

## Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Top docs**:

1. `medquad_ans_0000353-1` (score=0.6729) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control the amount of salt r...
2. `medquad_ans_0000353-2` (score=0.6665) - The prevalence of familial hyperaldosteronism is unknown. Familial hyperaldosteronism type II appears to be the most common variety. All types of familial hyperaldosteronism combined account for fewer than 1 out of 10 cases of hyperaldos...
3. `medquad_ans_0000353-5` (score=0.6503) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormone Health Network: A...
4. `medquad_ans_0005138-4` (score=0.6361) - How is pseudohypoaldosteronism type 2 diagnosed? Pseudohypoaldosteronism type 2 is usually diagnosed in adults. Unexplained hyperkalemia may be the presenting symptom and Pseudohypoaldosteronism type 2 may be diagnosed after common cause...
5. `medquad_ans_0002259-1` (score=0.6326) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is available, the tabl...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include severe hypertension that starts in childhood. The hypertension is difficult to treat and often results in damage to organs such as the heart and kidneys. Rarely, individuals with type III have milder symptoms with treatable hypertension and no adrenal gland enlargement [medquad_ans_0000353-1].

**Citations detected**: `medquad_ans_0000353-1`
**Supported citations**: `medquad_ans_0000353-1`
**Unsupported citations**: _none_

## Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Top docs**:

1. `medquad_ans_0005761-3` (score=0.6946) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation. Prenatal...
2. `medquad_ans_0000941-5` (score=0.6783) - These resources address the diagnosis or management of spinocerebellar ataxia type 36: - Ataxia Center at the University of Minnesota: Dominant Spinocerebellar Ataxias - Baylor College of Medicine: Parkinson's Disease Center and Movement...
3. `medquad_ans_0000942-5` (score=0.6691) - These resources address the diagnosis or management of SCA6: - Gene Review: Gene Review: Spinocerebellar Ataxia Type 6 - Genetic Testing Registry: Spinocerebellar ataxia 6 These resources from MedlinePlus offer information about the diag...
4. `medquad_ans_0000938-5` (score=0.6676) - These resources address the diagnosis or management of SCA1: - Gene Review: Gene Review: Spinocerebellar Ataxia Type 1 - Genetic Testing Registry: Spinocerebellar ataxia 1 These resources from MedlinePlus offer information about the diag...
5. `medquad_ans_0000032-3__dup1` (score=0.6661) - The prognosis for individuals with ataxia and cerebellar/spinocerebellar degeneration varies depending on its underlying cause.

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Citations detected**: `medquad_ans_0005761-3`
**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

## Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Top docs**:

1. `medquad_ans_0000503-2` (score=0.7001) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
2. `medquad_ans_0000503-5` (score=0.6739) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus offer information abou...
3. `medquad_ans_0000529-1` (score=0.6605) - Iron-refractory iron deficiency anemia is one of many types of anemia, which is a group of conditions characterized by a shortage of healthy red blood cells. This shortage prevents the blood from carrying an adequate supply of oxygen to ...
4. `medquad_ans_0000529-2` (score=0.6535) - Although iron deficiency anemia is relatively common, the prevalence of the iron-refractory form of the disease is unknown. At least 50 cases have been described in the medical literature. Researchers suspect that iron-refractory iron de...
5. `medquad_ans_0003040-1` (score=0.6487) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload. If the information ...

**Answer**:

The provided context does not specify the symptoms of Hypochromic microcytic anemia with iron overload. While it mentions that the condition involves hypochromic microcytic anemia and iron overload, and references the Human Phenotype Ontology for signs and symptoms, no explicit list or description of symptoms is given in the available documents [medquad_ans_0000503-2][medquad_ans_0000503-5][medquad_ans_0003040-1]. Therefore, based on the provided context, the symptoms cannot be determined.

**Citations detected**: `medquad_ans_0000503-2`, `medquad_ans_0000503-5`, `medquad_ans_0003040-1`
**Supported citations**: `medquad_ans_0000503-2`, `medquad_ans_0000503-5`, `medquad_ans_0003040-1`
**Unsupported citations**: _none_

## Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Top docs**:

1. `medquad_ans_0000115-2` (score=0.6861) - Benign chronic pemphigus is a rare condition; its prevalence is unknown.
2. `medquad_ans_0002290-1` (score=0.6522) - Familial pemphigus vulgaris refers to a cluster of pemphigus vulgaris within a family. Pemphigus vulgaris is a rare autoimmune condition that is characterized by blisters and sores on the skin and mucus membranes. Although the exact caus...
3. `medquad_ans_0000115-5` (score=0.6510) - These resources address the diagnosis or management of benign chronic pemphigus: - American Osteopathic College of Dermatology - Genetic Testing Registry: Familial benign pemphigus These resources from MedlinePlus offer information about...
4. `medquad_ans_0004796-1` (score=0.6458) - Pemphigus vulgaris is an autoimmune disorder that involves blistering of the skin and mucous membranes. It occurs almost exclusively in middle-aged or older people. Many cases begin with blisters in the mouth, followed by skin blisters t...
5. `medquad_ans_0000796-2` (score=0.6391) - PRS superactivity is believed to be a rare disorder. Approximately 30 families with the condition have been reported. More than two thirds of these families are affected by the milder form of the disease.

**Answer**:

The provided context does not indicate whether benign chronic pemphigus is inherited. While familial benign pemphigus is mentioned in the context and resources related to genetic testing are listed, there is no explicit information on the inheritance pattern of benign chronic pemphigus [medquad_ans_0000115-5]. Therefore, based on the available information, it cannot be determined if benign chronic pemphigus is inherited.

**Citations detected**: `medquad_ans_0000115-5`
**Supported citations**: `medquad_ans_0000115-5`
**Unsupported citations**: _none_

## Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Top docs**:

1. `medquad_ans_0000916-4__dup1` (score=0.7608) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is enough to cause fea...
2. `medquad_ans_0000916-5__dup1` (score=0.7454) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene known to be associa...
3. `medquad_ans_0000145-2` (score=0.7365) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
4. `medquad_ans_0000915-1__dup1` (score=0.7337) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
5. `medquad_ans_0000916-1__dup1` (score=0.7337) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause the condition. In some cases, the mutation is inherited from an affected parent, and in other cases, it occurs for the first time in the individual (de novo mutation) [medquad_ans_0000916-4__dup1]. The condition is caused by a mutation in the TGFB1 gene, which is inherited in an autosomal dominant fashion [medquad_ans_0000145-2].

**Citations detected**: `medquad_ans_0000916-4__dup1`, `medquad_ans_0000145-2`
**Supported citations**: `medquad_ans_0000916-4__dup1`, `medquad_ans_0000145-2`
**Unsupported citations**: _none_

## Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Top docs**:

1. `medquad_ans_0000483-2` (score=0.7179) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
2. `medquad_ans_0000580-4` (score=0.6983) - Langerhans cell histiocytosis is usually not inherited and typically occurs in people with no history of the disorder in their family. A few families with multiple cases of Langerhans cell histiocytosis have been identified, but the inhe...
3. `medquad_ans_0000324-4` (score=0.6977) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
4. `medquad_ans_0000483-5` (score=0.6954) - These resources address the diagnosis or management of histiocytosis-lymphadenopathy plus syndrome: - Genetic Testing Registry: Histiocytosis-lymphadenopathy plus syndrome These resources from MedlinePlus offer information about the diag...
5. `medquad_ans_0000580-3` (score=0.6923) - Somatic mutations in the BRAF gene have been identified in the Langerhans cells of about half of individuals with Langerhans cell histiocytosis. Somatic gene mutations are acquired during a person's lifetime and are present only in certa...

**Answer**:

The provided context does not contain information about the genetic changes related to histiocytosis-lymphadenopathy plus syndrome. Therefore, based on the available information, the genetic changes for this specific syndrome cannot be determined. [medquad_ans_0000483-2] and [medquad_ans_0000483-5] mention resources for diagnosis and management but do not specify genetic changes. [medquad_ans_0000580-3] discusses somatic mutations in the BRAF gene in Langerhans cell histiocytosis, which is a different condition. Thus, the context is insufficient to answer the question about genetic changes in histiocytosis-lymphadenopathy plus syndrome.

**Citations detected**: `medquad_ans_0000483-2`, `medquad_ans_0000483-5`, `medquad_ans_0000580-3`
**Supported citations**: `medquad_ans_0000483-2`, `medquad_ans_0000483-5`, `medquad_ans_0000580-3`
**Unsupported citations**: _none_

## Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7094) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0004450-3` (score=0.7006) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a pers...
3. `medquad_ans_0000738-5` (score=0.7001) - These resources address the diagnosis or management of Noonan syndrome: - Gene Review: Gene Review: Noonan Syndrome - Genetic Testing Registry: Noonan syndrome - Genetic Testing Registry: Noonan syndrome 1 - Genetic Testing Registry: Noo...
4. `medquad_ans_0000738-3` (score=0.6688) - Mutations in the PTPN11, SOS1, RAF1, KRAS, NRAS and BRAF genes cause Noonan syndrome. Most cases of Noonan syndrome result from mutations in one of three genes, PTPN11, SOS1, or RAF1. PTPN11 gene mutations account for approximately 50 pe...
5. `medquad_ans_0004453-1` (score=0.6541) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, congenital heart defect...

**Answer**:

Yes, Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a person with Noonan syndrome has a 50% (1 in 2) chance to inherit the condition. In some cases, the condition is inherited from an affected parent. In other cases, the condition is caused by a new mutation occurring for the first time in the affected person [medquad_ans_0004450-3].

**Citations detected**: `medquad_ans_0004450-3`
**Supported citations**: `medquad_ans_0004450-3`
**Unsupported citations**: _none_

## Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Top docs**:

1. `medquad_ans_0001093-2` (score=0.7538) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
2. `medquad_ans_0001093-5` (score=0.7082) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Acidosis These resource...
3. `medquad_ans_0001092-5` (score=0.7028) - These resources address the diagnosis or management of 3-MCC deficiency: - Baby's First Test - Genetic Testing Registry: 3 Methylcrotonyl-CoA carboxylase 1 deficiency - Genetic Testing Registry: 3-methylcrotonyl CoA carboxylase 2 deficie...
4. `medquad_ans_0001089-2` (score=0.7022) - HMG-CoA lyase deficiency is a rare condition; it has been reported in fewer than 100 individuals worldwide. Most people diagnosed with this disorder have been from Saudi Arabia, Portugal, or Spain.
5. `medquad_ans_0000155-3__dup1` (score=0.7002) - How might adenylosuccinase deficiency be treated? At the current time, there are no effective therapies for the treatment of adenylosuccinase deficiency. Treatment is supportive based on the specific features.

**Answer**:

The provided context does not contain information about the treatments for 3-methylglutaconyl-CoA hydratase deficiency. While it mentions resources related to the diagnosis and management of the condition, such as genetic testing and diagnostic tests, it does not specify any treatments [medquad_ans_0001093-5]. Therefore, based on the available information, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

**Citations detected**: `medquad_ans_0001093-5`
**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

## Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.7197) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000505-2` (score=0.6724) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
3. `medquad_ans_0003059-1` (score=0.6585) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles in the arms and legs....
4. `medquad_ans_0003059-5` (score=0.6578) - How is hypokalemic periodic paralysis diagnosed? The diagnosis of hypokalemic periodic paralysis is based on a history of episodes of paralysis and low levels of potassium in the blood during attacks (less than 0.9 to 3.0 mmol/L), but no...
5. `medquad_ans_0003059-4` (score=0.6555) - How is hypokalemic periodic paralysis inherited? This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person h...

**Answer**:

The provided context does not contain information about hyperkalemic periodic paralysis. Therefore, it is not possible to answer the question based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.6649) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of extraocular muscles...
2. `medquad_ans_0000225-1` (score=0.6365) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Congenital fibrosis of t...
3. `medquad_ans_0000054-2__dup1` (score=0.6116) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
4. `medquad_ans_0000225-2` (score=0.6049) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few families of Turkish, Sau...
5. `medquad_ans_0000225-4` (score=0.6033) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suff...

**Answer**:

The provided context does not specify detailed treatments for congenital fibrosis of the extraocular muscles. It mentions that resources from MedlinePlus cover diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, which may be relevant to management [medquad_ans_0000225-5]. However, no explicit information is given about specific treatments for congenital fibrosis of the extraocular muscles. Therefore, based on the available context, a definitive answer cannot be provided.

Answer: Insufficient context to determine specific treatments for congenital fibrosis of the extraocular muscles. [medquad_ans_0000225-5]

**Citations detected**: `medquad_ans_0000225-5`
**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

## Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Top docs**:

1. `medquad_ans_0000152-3__dup1` (score=0.7414) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the course of HD.
2. `medquad_ans_0000489-2` (score=0.7160) - Huntington disease affects an estimated 3 to 7 per 100,000 people of European ancestry. The disorder appears to be less common in some other populations, including people of Japanese, Chinese, and African descent.
3. `medquad_ans_0000489-4` (score=0.7104) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. An affected person usually inherits the altered gene from one affected parent. In ra...
4. `medquad_ans_0000075-3__dup1` (score=0.7081) - The prognosis for individuals with chorea varies depending on the type of chorea and the associated disease. Huntington's disease is a progressive, and ultimately, fatal disease. Syndenham's chorea is treatable and curable.
5. `medquad_ans_0002944-4` (score=0.7072) - How is Huntington disease inherited? Huntington disease (HD) is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one of the 2 copies of the HTT gene is enough to cause the condition. When a pe...

**Answer**:

Huntington's disease is a progressive, ultimately fatal disorder that causes disability worsening over time. It is inherited in an autosomal dominant pattern, meaning one copy of the altered HTT gene in each cell is sufficient to cause the disorder. Individuals typically inherit the altered gene from an affected parent, though in rare cases, the disease results from a new (de novo) mutation with no family history. The size of the CAG trinucleotide repeat in the HTT gene often increases from one generation to the next, a phenomenon known as anticipation. A larger number of repeats is associated with an earlier onset of symptoms. People with the adult-onset form typically have 40 to 50 CAG repeats, while those with the juvenile form have more than 60 repeats. Individuals with 27 to 35 CAG repeats do not develop the disease but are at risk of passing it to their children. The disease affects an estimated

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7619) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
2. `medquad_ans_0000416-3` (score=0.7368) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
3. `medquad_ans_0000416-5` (score=0.7174) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlinePlus offer informatio...
4. `medquad_ans_0000442-3` (score=0.7076) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step production (synthesis)...
5. `medquad_ans_0002603-1` (score=0.7045) - Glutaric acidemia type I (GA1) is an inherited disorder in which the body can't process certain proteins properly. People with GA1 have inadequate levels of an enzyme needed to break down certain amino acids. These amino acids and their ...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Citations detected**: `medquad_ans_0000416-2`
**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

## Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7300) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0004450-3` (score=0.7173) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a pers...
3. `medquad_ans_0000738-3` (score=0.6991) - Mutations in the PTPN11, SOS1, RAF1, KRAS, NRAS and BRAF genes cause Noonan syndrome. Most cases of Noonan syndrome result from mutations in one of three genes, PTPN11, SOS1, or RAF1. PTPN11 gene mutations account for approximately 50 pe...
4. `medquad_ans_0000738-5` (score=0.6928) - These resources address the diagnosis or management of Noonan syndrome: - Gene Review: Gene Review: Noonan Syndrome - Genetic Testing Registry: Noonan syndrome - Genetic Testing Registry: Noonan syndrome 1 - Genetic Testing Registry: Noo...
5. `medquad_ans_0000245-3` (score=0.6883) - Mutations in the HRAS gene cause Costello syndrome. This gene provides instructions for making a protein called H-Ras, which is part of a pathway that helps control cell growth and division. Mutations that cause Costello syndrome lead to...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Citations detected**: `medquad_ans_0000738-2`
**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: _none_

## Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Top docs**:

1. `medquad_ans_0000339-1__dup3` (score=0.7672) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels of society.
2. `medquad_ans_0000339-5__dup1` (score=0.7244) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, e...
3. `medquad_ans_0000557-1__dup2` (score=0.7181) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice that live on humans a...
4. `medquad_ans_0000339-2__dup1` (score=0.7136) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasionally pubic lice may be...
5. `medquad_ans_0000339-7` (score=0.6981) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to help prevent and con...

**Answer**:

Pubic "crab" lice infestation is diagnosed by finding a "crab" louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp). Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Citations detected**: `medquad_ans_0000339-5__dup1`
**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

## Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.7533) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of the disease, and other...
2. `medquad_ans_0005571-1` (score=0.7315) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration and frequency of the...
3. `medquad_ans_0000275-2` (score=0.7279) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United States . Treatments are...
4. `medquad_ans_0001036-2` (score=0.7160) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
5. `medquad_ans_0000860-2__dup1` (score=0.7157) - How might Brown-Sequard syndrome be treated?

**Answer**:

Treatment options that have been attempted for Shapiro syndrome include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. However, these treatments have resulted in varying responses, and some have been unsuccessful or of doubtful efficacy due to the small number of diagnosed cases [medquad_ans_0005571-3]. The cause of Shapiro syndrome is unknown, and thus, there is no established standard treatment. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Citations detected**: `medquad_ans_0005571-3`
**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

## Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Top docs**:

1. `medquad_ans_0000373-2` (score=0.7116) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen with other disorders.
2. `medquad_ans_0002359-3` (score=0.6960) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiologists, surgeons, gast...
3. `medquad_ans_0000373-5` (score=0.6670) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testing Registry: FG synd...
4. `medquad_ans_0000373-4` (score=0.6628) - FG syndrome is inherited in an X-linked recessive pattern. The genes likely associated with this disorder, including MED12, are located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosom...
5. `medquad_ans_0000373-1` (score=0.6623) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects intelligence and beh...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects intelligence and behavior, with almost everyone having intellectual disability that ranges from mild to severe. Affected individuals tend to be friendly, inquisitive, and hyperactive, with a short attention span. Compared to people with other forms of intellectual disability, their socialization and daily living skills are strong, while verbal skills may be impaired. The syndrome is inherited in an X-linked recessive pattern, with genes likely associated with it, including MED12, located on the X chromosome. Males are more frequently affected than females because they have only one X chromosome, and a single altered copy of the gene is sufficient to cause the condition [medquad_ans_0000373-1, medquad_ans_0000373-

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Top docs**:

1. `medquad_ans_0000634-2` (score=0.6981) - Meckel syndrome affects 1 in 13,250 to 1 in 140,000 people worldwide. It is more common in certain populations; for example, the condition affects about 1 in 9,000 people of Finnish ancestry and about 1 in 3,000 people of Belgian ancestry.
2. `medquad_ans_0001019-2` (score=0.6813) - Vohwinkel syndrome is a rare disorder; about 50 cases have been reported in the medical literature.
3. `medquad_ans_0000565-2` (score=0.6744) - Knobloch syndrome is a rare condition. However, the exact prevalence of the condition is unknown.
4. `medquad_ans_0000433-2` (score=0.6694) - Gorlin syndrome affects an estimated 1 in 31,000 people. While more than 1 million new cases of basal cell carcinoma are diagnosed each year in the United States, fewer than 1 percent of these skin cancers are related to Gorlin syndrome.
5. `medquad_ans_0000982-2` (score=0.6654) - Tietz syndrome is a rare disorder; its exact prevalence is unknown. Only a few affected families have been described in the medical literature.

**Answer**:

The provided context does not contain any information about the symptoms of Tukel syndrome. Therefore, it is not possible to answer the question based on the given text. [Note: There may be a typo in the query, as "Tukel syndrome" is not mentioned in the context; the relevant syndromes discussed are Meckel, Vohwinkel, Knobloch, Gorlin, and Tietz.]

**Citations detected**: `Note: There may be a typo in the query, as "Tukel syndrome" is not mentioned in the context; the relevant syndromes discussed are Meckel, Vohwinkel, Knobloch, Gorlin, and Tietz.`
**Supported citations**: _none_
**Unsupported citations**: `Note: There may be a typo in the query, as "Tukel syndrome" is not mentioned in the context; the relevant syndromes discussed are Meckel, Vohwinkel, Knobloch, Gorlin, and Tietz.`

## Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.7007) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000675-2` (score=0.6946) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
3. `medquad_ans_0000674-2` (score=0.6941) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000676-2` (score=0.6872) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000677-5` (score=0.6720) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastrin These resources f...

**Answer**:

The provided context does not specify any treatments for mucolipidoses. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and genetic testing registry, no specific treatments are detailed for mucolipidosis types IV, III, or II [medquad_ans_0000677-5]. Therefore, based on the given information, the treatments for mucolipidoses cannot be determined.

**Citations detected**: `medquad_ans_0000677-5`
**Supported citations**: `medquad_ans_0000677-5`
**Unsupported citations**: _none_

## Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Top docs**:

1. `medquad_ans_0000829-5` (score=0.6490) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholestasis 2 - Genetic Tes...
2. `medquad_ans_0000118-5` (score=0.6457) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 - Genetic Testing Reg...
3. `medquad_ans_0005084-1` (score=0.6275) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching, jaundice, failure ...
4. `medquad_ans_0000181-13` (score=0.6272) - - Alagille syndrome is a genetic condition that results in various symptoms in different parts of the body, including the liver. - A person with Alagille syndrome has fewer than the normal number of small bile ducts inside the liver. - I...
5. `medquad_ans_0005084-2` (score=0.6253) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis type 2. If the info...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 (PFIC2) include severe itching, jaundice, failure to thrive, portal hypertension (high blood pressure in the vein that provides blood to the liver), and hepatosplenomegaly (enlarged liver and spleen) [medquad_ans_0005084-1].

**Citations detected**: `medquad_ans_0005084-1`
**Supported citations**: `medquad_ans_0005084-1`
**Unsupported citations**: _none_

## Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.7128) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he is more likely to dev...
2. `medquad_ans_0000070-7` (score=0.6871) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is help. Urinary incont...
3. `medquad_ans_0000211-1__dup3` (score=0.6718) - Urinary incontinence is the loss of bladder control, resulting in the accidental leakage of urine from the body. For example, a man may feel a strong, sudden need, or urgency, to urinate just before losing a large amount of urine, called...
4. `medquad_ans_0000931-1__dup2` (score=0.6708) - Urinary incontinence (UI) is loss of bladder control. Symptoms can range from mild leaking to uncontrollable wetting. It can happen to anyone, but it becomes more common with age. Women experience UI twice as often as men. Most bladder c...
5. `medquad_ans_0000219-4__dup3` (score=0.6673) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men age 60 to 69 - 30 perc...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6].

**Citations detected**: `medquad_ans_0000211-6`
**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

## Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0005415-3` (score=0.6768) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell must have a mutation...
2. `medquad_ans_0000882-5` (score=0.6710) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Registry: Robinow syndro...
3. `medquad_ans_0005415-4` (score=0.6605) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of Robinow syndrome have...
4. `medquad_ans_0000882-4` (score=0.6570) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an ind...
5. `medquad_ans_0000882-2` (score=0.6484) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the condition has been id...

**Answer**:

The provided context does not define what Robinow syndrome is in general terms. It describes the inheritance patterns (autosomal recessive or autosomal dominant), genetic testing availability, and resources for diagnosis and management, but does not provide a definition of Robinow syndrome itself. Therefore, based on the given information, it is not possible to answer what Robinow syndrome is.

[medquad_ans_0005415-3] [medquad_ans_0000882-5] [medquad_ans_0005415-4]

**Citations detected**: `medquad_ans_0005415-3`, `medquad_ans_0000882-5`, `medquad_ans_0005415-4`
**Supported citations**: `medquad_ans_0005415-3`, `medquad_ans_0000882-5`, `medquad_ans_0005415-4`
**Unsupported citations**: _none_

## Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0000551-2` (score=0.7002) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
2. `medquad_ans_0003331-3` (score=0.6875) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...
3. `medquad_ans_0003331-4` (score=0.6858) - How is juvenile primary lateral sclerosis inherited? Juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an...
4. `medquad_ans_0003331-1` (score=0.6806) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
5. `medquad_ans_0000551-4` (score=0.6803) - When caused by mutations in the ALS2 gene, juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress over a period of 15 to 20 years. It is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs and symptoms [medquad_ans_0003331-1]. Mutations in the ALS2 gene alter the instructions for producing alsin, a protein abundant in motor neurons, leading to an unstable or nonfunctional alsin protein, which is believed to damage motor neurons and cause the disorder [medquad_ans_000

**Citations detected**: `medquad_ans_0003331-1`
**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

## Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Top docs**:

1. `medquad_ans_0000040-10` (score=0.7270) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
2. `medquad_ans_0000040-18` (score=0.7080) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certain features of the le...
3. `medquad_ans_0000554-1__dup2` (score=0.7054) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells crowd out the healthy...
4. `medquad_ans_0000267-2` (score=0.6970) - Acute myeloid leukemia occurs in approximately 3.5 per 100,000 individuals each year. Forty to 50 percent of people with acute myeloid leukemia have CN-AML.
5. `medquad_ans_0000040-19` (score=0.6958) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemia. Learn more about ...

**Answer**:

Leukemia is a cancer of the white blood cells. White blood cells help the body fight infection, and they form in the bone marrow. In leukemia, the bone marrow produces abnormal white blood cells that crowd out healthy blood cells, impairing the blood's ability to function properly. There are different types of leukemia, including acute lymphocytic leukemia, acute myeloid leukemia, chronic lymphocytic leukemia, and chronic myeloid leukemia. Leukemia can develop quickly or slowly; chronic leukemia grows slowly, while acute leukemia involves rapid increases in very abnormal cells. Adults can develop either type, though children most often have an acute form [medquad_ans_0000554-1__dup2].

**Citations detected**: `medquad_ans_0000554-1__dup2`
**Supported citations**: `medquad_ans_0000554-1__dup2`
**Unsupported citations**: _none_

## Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Top docs**:

1. `medquad_ans_0002485-2` (score=0.6705) - What are the signs and symptoms of gangliocytomas? Signs and symptoms caused by the presence of a gangliocytoma can vary depending on the tumor's location. Seizures are the most common symptom. Other symptoms may include increased brain ...
2. `medquad_ans_0000006_7-3` (score=0.6449) - The cause of most childhood CNS germ cell tumors is not known.
3. `medquad_ans_0000326-1__dup1` (score=0.6429) - Anaplastic ganglioglioma (AGG) is a very rare type of brain tumor that is a type of ganglioglioma. In general, gangliogliomas are classified as grade I or low grade tumors, meaning that they grow slowly and are considered benign. Anaplas...
4. `medquad_ans_0001809-3` (score=0.6412) - How are desmoplastic infantile gangliomas diagnosed? In addition to detecting the signs and symptoms commonly seen in DIGs, head CT scans and MRIs may reveal the presence of this type of brain tumor.
5. `medquad_ans_0001809-4` (score=0.6351) - What treatment is available for desmoplastic infantile gangliomas? Surgical resection (removal of the area of the brain with the tumor) has been the standard treatment reported in the medical literature. The size of the resection is prob...

**Answer**:

Gangliocytomas are brain tumors that can present with a variety of signs and symptoms depending on their location. The most common symptom is seizures. Other possible symptoms include increased brain pressure, endocrine disorders, and focal symptoms. In some cases, gangliocytomas are asymptomatic and may be diagnosed incidentally on imaging studies [medquad_ans_0002485-2].

Note: The provided context does not specify the exact nature or classification of gangliocytomas beyond their symptoms and diagnostic presentation. It is important to distinguish that gangliocytomas are different from gangliogliomas, which are a related but distinct type of tumor [medquad_ans_0000326-1__dup1]. However, no direct definition or classification of gangliocytomas is given in the context.

**Citations detected**: `medquad_ans_0002485-2`, `medquad_ans_0000326-1__dup1`
**Supported citations**: `medquad_ans_0002485-2`, `medquad_ans_0000326-1__dup1`
**Unsupported citations**: _none_

## Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Top docs**:

1. `medquad_ans_0000130-3__dup1` (score=0.7401) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are left with residual defi...
2. `medquad_ans_0000130-4__dup1` (score=0.7398) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
3. `medquad_ans_0000658-2` (score=0.7247) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
4. `medquad_ans_0000130-1__dup1` (score=0.7170) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of the tendon reflexes....
5. `medquad_ans_0000130-2__dup1` (score=0.6986) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportive care.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of the tendon reflexes. Like Guillain-Barr syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1].

**Citations detected**: `medquad_ans_0000130-1__dup1`
**Supported citations**: `medquad_ans_0000130-1__dup1`
**Unsupported citations**: _none_

## Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Top docs**:

1. `medquad_ans_0001084-2` (score=0.7085) - 22q11.2 deletion syndrome affects an estimated 1 in 4,000 people. However, the condition may actually be more common than this estimate because doctors and researchers suspect it is underdiagnosed due to its variable features. The condit...
2. `medquad_ans_0006570-4` (score=0.6923) - Is 22q11.2 deletion syndrome inherited? Most cases of 22q11.2 deletion syndrome are not inherited from a parent and are caused by a random error during the formation of egg or sperm cells, or during early fetal development. In about 10% ...
3. `medquad_ans_0001084-4` (score=0.6764) - The inheritance of 22q11.2 deletion syndrome is considered autosomal dominant because a deletion in one copy of chromosome 22 in each cell is sufficient to cause the condition. Most cases of 22q11.2 deletion syndrome are not inherited, h...
4. `medquad_ans_0001086-2` (score=0.6742) - At least 500 cases of 22q13.3 deletion syndrome are known.
5. `medquad_ans_0001085-5` (score=0.6638) - These resources address the diagnosis or management of 22q11.2 duplication: - Gene Review: Gene Review: 22q11.2 Duplication - Genetic Testing Registry: 22q11.2 duplication syndrome These resources from MedlinePlus offer information about...

**Answer**:

22q11.2 deletion syndrome is caused by a deletion of genetic material in a region of chromosome 22, specifically at band 22q11.2. Most cases are not inherited and result from a random error during the formation of egg or sperm cells or during early fetal development. In about 10% of cases, the deletion is inherited from a parent who has the deletion. The condition follows an autosomal dominant inheritance pattern because having a deletion in only one copy of chromosome 22 in each cell is sufficient to cause the signs and symptoms [medquad_ans_0006570-4][medquad_ans_0001084-4].

**Citations detected**: `medquad_ans_0006570-4`, `medquad_ans_0001084-4`
**Supported citations**: `medquad_ans_0006570-4`, `medquad_ans_0001084-4`
**Unsupported citations**: _none_

## Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Top docs**:

1. `medquad_ans_0000062-8` (score=0.7221) - Shingles is very common. Fifty percent of all Americans will have had shingles by the time they are 80. While shingles occurs in people of all ages, it is most common in 60-to 80-year-olds. In fact, one out of every three people 60 years...
2. `medquad_ans_0000812-1__dup2` (score=0.7117) - Shingles is a disease caused by the varicella-zoster virus - the same virus that causes chickenpox. After you have chickenpox, the virus stays in your body. It may not cause problems for many years. As you get older, the virus may reappe...
3. `medquad_ans_0000062-12` (score=0.6994) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained. It is with you for ...
4. `medquad_ans_0000149-4__dup1` (score=0.6983) - The NINDS supports research on shingles and shingles-related conditions. Current studies focus on the relationship between the persistence of neurotropic viruses and development of neurological diseases including herpes simplex and varic...
5. `medquad_ans_0000148-1__dup2` (score=0.6879) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain (which can be severe...

**Answer**:

Shingles is caused by the varicella-zoster virus -- the same virus that causes chickenpox. After a person recovers from chickenpox, the virus remains in the body for life and can become active again, especially in later years when the immune system is less effective. The virus travels from spinal nerve cells along a nerve path to the skin, causing inflammation of nerve endings and a painful rash [medquad_ans_0000062-12]. [medquad_ans_0000812-1__dup2]

**Citations detected**: `medquad_ans_0000062-12`, `medquad_ans_0000812-1__dup2`
**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000812-1__dup2`
**Unsupported citations**: _none_

## Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Top docs**:

1. `medquad_ans_0000333-2` (score=0.6904) - Fabry disease affects an estimated 1 in 40,000 to 60,000 males. This disorder also occurs in females, although the prevalence is unknown. Milder, late-onset forms of the disorder are probably more common than the classic, severe form.
2. `medquad_ans_0000123-3__dup1` (score=0.6895) - Individuals with Fabry disease often die prematurely of complications from strokes, heart disease, or kidney failure.
3. `medquad_ans_0000392-4` (score=0.6841) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual with an autosomal rece...
4. `medquad_ans_0000038-2` (score=0.6824) - The prevalence of Alpers-Huttenlocher syndrome is approximately 1 in 100,000 individuals.
5. `medquad_ans_0002458-3` (score=0.6776) - How is Fryns syndrome inherited? Although the exact cause of Fryns syndrome is not currently known (and no disease-causing gene has yet been identified), it is thought to be genetic because it tends to "run in families" and has features ...

**Answer**:

The provided context does not mention symptoms of "Fryns Hofkens Fabry syndrome." This condition does not appear to be a recognized or described medical entity in the given information. The context separately discusses Fabry disease and Fryns syndrome, but no symptoms are provided for a combined or incorrect condition named "Fryns Hofkens Fabry syndrome." Therefore, based on the available information, the symptoms cannot be determined.

Note: There may be a confusion in the name of the condition. Fabry disease and Fryns syndrome are distinct disorders. The context provides no information on symptoms of a combined or erroneous condition. [medquad_ans_0000333-2], [medquad_ans_0000123-3__dup1], [medquad_ans_0000392-4], and [medquad_ans_0002458-3] discuss Fabry

**Citations detected**: `medquad_ans_0000333-2`, `medquad_ans_0000123-3__dup1`, `medquad_ans_0000392-4`, `medquad_ans_0002458-3`
**Supported citations**: `medquad_ans_0000333-2`, `medquad_ans_0000123-3__dup1`, `medquad_ans_0000392-4`, `medquad_ans_0002458-3`
**Unsupported citations**: _none_

## Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.7539) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research focused on - determinin...
2. `medquad_ans_0000050-16` (score=0.7353) - Women have smaller bones, and they lose bone more rapidly than men because of hormone changes that occur after menopause. Therefore, women are at higher risk for osteoporosis.
3. `medquad_ans_0000050-10` (score=0.7346) - Osteoporosis is a disease that thins and weakens the bones to the point that they break easily. Women and men with osteoporosis most often break bones in the hip, spine, and wrist, but osteoporosis can be the cause of bone fractures anyw...
4. `medquad_ans_0000552-2` (score=0.7269) - The prevalence of juvenile primary osteoporosis is unknown. Nearly 1 in 10 adults over age 50 have osteoporosis, but the condition is uncommon in children. Osteoporosis can occur at a young age as a feature of other conditions but rarely...
5. `medquad_ans_0000050-27` (score=0.7124) - Osteoporosis does not have any symptoms until a fracture occurs. Some people may be unaware that they have already experienced one or more spine fractures. Height loss of one inch or more may be the first sign that someone has experience...

**Answer**:

Scientists are pursuing a wide range of basic and clinical studies on osteoporosis, focusing on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of factors such as genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. Additional information about ongoing research can be obtained from the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH [medquad_ans_0000050-8].

**Citations detected**: `medquad_ans_0000050-8`
**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

## Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Top docs**:

1. `medquad_ans_0000338-1__dup1` (score=0.7015) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the source. If you use th...
2. `medquad_ans_0000869-1__dup1` (score=0.6934) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can bring: - Lists of your c...
3. `medquad_ans_0000694-1__dup1` (score=0.6918) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of medical errors, you ca...
4. `medquad_ans_0000067-17` (score=0.6917) - It is important to keep a copy of your medical records to share with any new doctors that you see. This information should contain the type of cancer you were diagnosed with, test results, and treatment details. It is also essential to i...
5. `medquad_ans_0000430-1__dup1` (score=0.6849) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. At worst, they're da...

**Answer**:

Yes, there is information about health fraud. The context mentions that health fraud involves selling drugs, devices, foods, or cosmetics that are not safe or effective [medquad_ans_0000430-1__dup1]. However, the provided summary does not include further details on how to identify or report such fraud.

**Citations detected**: `medquad_ans_0000430-1__dup1`
**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

## Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Top docs**:

1. `medquad_ans_0000055-3__dup2` (score=0.7230) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision - dark or floating sp...
2. `medquad_ans_0000057-1__dup5` (score=0.7119) - Too much glucose in the blood for a long time can cause diabetes problems. This high blood glucose, also called blood sugar, can damage many parts of the body, such as the heart, blood vessels, eyes, and kidneys. Heart and blood vessel d...
3. `medquad_ans_0000343-1__dup1` (score=0.7027) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage to the optic nerve - ...
4. `medquad_ans_0000275-1__dup1` (score=0.7015) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adults. Your retina is t...
5. `medquad_ans_0000022-4__dup3` (score=0.6996) - The signs and symptoms of type 2 diabetes can be so mild that you might not even notice them. Nearly 7 million people in the United States have type 2 diabetes and dont know they have the disease. Many have no signs or symptoms. Some peo...

**Answer**:

The provided context does not specify symptoms related to "Prevent diabetes problems: Keep your eyes healthy." While it outlines symptoms of diabetic eye problems such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes [medquad_ans_0000055-3__dup2], these are symptoms of existing eye issues, not preventive measures or symptoms to prevent diabetes problems.

Therefore, based on the given context, there is no direct information about symptoms associated with preventing diabetes problems through eye health maintenance.

Answer: Insufficient context to determine symptoms for "Prevent diabetes problems: Keep your eyes healthy." [medquad_ans_0000055-3__dup2]

**Citations detected**: `medquad_ans_0000055-3__dup2`
**Supported citations**: `medquad_ans_0000055-3__dup2`
**Unsupported citations**: _none_

## Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000001-2` (score=0.7485) - Aarskog-Scott syndrome is believed to be a rare disorder; however, its prevalence is unknown because mildly affected people are often not diagnosed.
2. `medquad_ans_0000001-3` (score=0.7116) - Mutations in the FGD1 gene cause some cases of Aarskog-Scott syndrome. The FGD1 gene provides instructions for making a protein that turns on (activates) another protein called Cdc42, which transmits signals that are important for variou...
3. `medquad_ans_0000001-4` (score=0.6865) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered...
4. `medquad_ans_0000540-2` (score=0.6794) - Jackson-Weiss syndrome is a rare genetic disorder; its incidence is unknown.
5. `medquad_ans_0000392-4` (score=0.6785) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual with an autosomal rece...

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Citations detected**: `medquad_ans_0000001-4`
**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

## Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Top docs**:

1. `medquad_ans_0001063-4` (score=0.6924) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
2. `medquad_ans_0001062-4` (score=0.6905) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
3. `medquad_ans_0001063-2` (score=0.6761) - X-linked sideroblastic anemia and ataxia is a rare disorder; only a few affected families have been reported.
4. `medquad_ans_0001063-1` (score=0.6710) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anemia results when deve...
5. `medquad_ans_0000369-4` (score=0.6709) - Fanconi anemia is most often inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the muta...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ABCB7 gene, which is located on the X chromosome. In males, one altered copy of the gene in each cell is sufficient to cause the condition. Females typically require mutations in both copies of the gene to be affected, making the disorder much more common in males. Carriers of an ABCB7 mutation do not develop ataxia or other health problems but may have abnormally small, pale red blood cells detectable via blood tests [medquad_ans_0001063-4].

Additionally, mutations in the ALAS2 gene are associated with X-linked sideroblastic anemia, though this condition is more commonly linked to ABCB7 [medquad_ans_0001062-4]. However, the primary genetic change associated with X-linked sideroblastic anemia and ataxia is the ABCB7

**Citations detected**: `medquad_ans_0001063-4`, `medquad_ans_0001062-4`
**Supported citations**: `medquad_ans_0001063-4`, `medquad_ans_0001062-4`
**Unsupported citations**: _none_

## Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0003601-1` (score=0.7033) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of years. Onset may occu...
2. `medquad_ans_0000602-2` (score=0.6991) - It is difficult to determine the prevalence of limb-girdle muscular dystrophy because its features vary and overlap with those of other muscle disorders. Prevalence estimates range from 1 in 14,500 to 1 in 123,000 individuals.
3. `medquad_ans_0003610-1` (score=0.6863) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdle muscular dystrophi...
4. `medquad_ans_0003601-2` (score=0.6851) - How is limb-girdle muscular dystrophy inherited? Limb-girdle muscular dystrophy (LGMD) is most often inherited in an autosomal recessive manner; less commonly, rare sub-types may be inherited in an autosomal dominant manner. There may be...
5. `medquad_ans_0000602-4` (score=0.6720) - Limb-girdle muscular dystrophy can have different inheritance patterns. Most forms of this condition are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an indi...

**Answer**:

Limb-girdle muscular dystrophy is a group of disorders that affect the voluntary muscles around the hips and shoulders, characterized by progressive loss of muscle strength and bulk over time. The condition may onset in childhood, adolescence, young adulthood, or later in life, and affects males and females equally. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern. There are at least 20 different types of limb-girdle muscular dystrophy [medquad_ans_0003601-1]. LGMD2C, a specific type, is caused by mutations in the gamma-sarcoglycan gene and is inherited in an autosomal recessive manner [medquad_ans_0003610-1]. While there are no treatments that directly reverse muscle weakness, supportive care can help reduce complications [medquad_ans_000360

**Citations detected**: `medquad_ans_0003601-1`, `medquad_ans_0003610-1`
**Supported citations**: `medquad_ans_0003601-1`, `medquad_ans_0003610-1`
**Unsupported citations**: _none_

## Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Top docs**:

1. `medquad_ans_0000055-3__dup2` (score=0.6608) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision - dark or floating sp...
2. `medquad_ans_0000016-6` (score=0.6510) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
3. `medquad_ans_0000343-1__dup1` (score=0.6387) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage to the optic nerve - ...
4. `medquad_ans_0000042-7` (score=0.6370) - Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-related macular degenerat...
5. `medquad_ans_0000055-7` (score=0.6280) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need other treatments. You wi...

**Answer**:

Diabetic eye problems are a type of retinal disorder that occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. In the early stages, no symptoms may appear, but as the condition worsens, symptoms can include blurry or double vision, rings, flashing lights, or blank spots in vision, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes [medquad_ans_0000055-3__dup2]. Diabetic retinopathy, a common form of diabetic eye problem, is a leading cause of blindness and usually affects both eyes [medquad_ans_0000016-6]. These problems can lead to permanent vision loss if not detected and treated early, emphasizing the importance of regular eye checkups [medquad_ans_0000343-1__dup1].

**Citations detected**: `medquad_ans_0000055-3__dup2`, `medquad_ans_0000016-6`, `medquad_ans_0000343-1__dup1`
**Supported citations**: `medquad_ans_0000055-3__dup2`, `medquad_ans_0000016-6`, `medquad_ans_0000343-1__dup1`
**Unsupported citations**: _none_

## Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.7908) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0006449-5` (score=0.7501) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, whi...
3. `medquad_ans_0000276-4__dup1` (score=0.7489) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct and/or support rese...
4. `medquad_ans_0006449-3` (score=0.7482) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body. It also helps remo...
5. `medquad_ans_0000670-2` (score=0.7440) - The prevalence of Mowat-Wilson syndrome is unknown. More than 200 people with this condition have been reported in the medical literature.

**Answer**:

Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals. It is caused by changes (mutations) in the ATP7B gene, which encodes a protein responsible for transporting copper from the liver to the rest of the body and removing excess copper from the body. Mutations in this gene prevent the protein from functioning properly, leading to an accumulation of copper in the body. Because high levels of copper are toxic, this buildup can damage tissues and organs and cause the signs and symptoms of Wilson disease [medquad_ans_0003769-4__dup1].

**Citations detected**: `medquad_ans_0003769-4__dup1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003769-4__dup1`

## Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Top docs**:

1. `medquad_ans_0000307-2` (score=0.6897) - Dyserythropoietic anemia and thrombocytopenia is a rare condition; its prevalence is unknown. Occasionally, individuals with this disorder are mistakenly diagnosed as having more common blood disorders, making it even more difficult to d...
2. `medquad_ans_0000283-2` (score=0.6892) - Diamond-Blackfan anemia affects approximately 5 to 7 per million liveborn infants worldwide.
3. `medquad_ans_0001503-3` (score=0.6888) - How might congenital dyserythropoietic anemia (CDA) type 2 be treated? The goal of CDA type 2 treatment is to address and prevent complications from anemia and iron overload. Most people with CDA type 2 develop iron overload, for some th...
4. `medquad_ans_0001062-2` (score=0.6787) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
5. `medquad_ans_0000307-1` (score=0.6766) - Dyserythropoietic anemia and thrombocytopenia is a condition that affects blood cells and primarily occurs in males. A main feature of this condition is a type of anemia called dyserythropoietic anemia, which is characterized by a shorta...

**Answer**:

The provided context does not describe the symptoms of Congenital dyserythropoietic anemia type 2. While it discusses treatment options and complications such as anemia and iron overload, specific symptoms are not mentioned [medquad_ans_0001503-3]. Therefore, based on the given information, the symptoms cannot be determined.

**Citations detected**: `medquad_ans_0001503-3`
**Supported citations**: `medquad_ans_0001503-3`
**Unsupported citations**: _none_

## Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000033-1__dup4` (score=0.7259) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage kidney disease or ES...
2. `medquad_ans_0000155-3__dup3` (score=0.7256) - Chronic kidney disease means you have damaged kidneys that cannot filter blood normally. Wastes then build up in your blood, harming your body. Kidney disease usually does not get better and may lead to kidney failure. If your kidneys fa...
3. `medquad_ans_0000038-33` (score=0.7139) - There are many researchers who are working on kidney disease. They are looking for ways to improve diagnosis, make treatments more effective, and make dialysis and transplantation work better. Several areas of research supported by the N...
4. `medquad_ans_0000058-11__dup1` (score=0.7118) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
5. `medquad_ans_0001009-2` (score=0.7094) - The prevalence of uromodulin-associated kidney disease is unknown. It accounts for fewer than 1 percent of cases of kidney disease.

**Answer**:

Kidney disease refers to conditions in which the kidneys are damaged and cannot filter blood normally, leading to the buildup of wastes in the body [medquad_ans_0000155-3__dup3]. Diabetic kidney disease, also known as diabetic nephropathy, is a specific type of kidney disease caused by diabetes [medquad_ans_0000033-1__dup4]. Chronic kidney disease (CKD) is a condition characterized by damaged kidneys that fail to filter blood effectively, and it may progress to kidney failure [medquad_ans_0000155-3__dup3]. Kidney failure, or end-stage kidney disease (ESRD), occurs when the kidneys no longer work well enough to perform their essential functions, requiring treatment such as dialysis or a kidney transplant [medquad_ans_0000058-11__dup1]. Diabetes is the

**Citations detected**: `medquad_ans_0000155-3__dup3`, `medquad_ans_0000033-1__dup4`, `medquad_ans_0000058-11__dup1`
**Supported citations**: `medquad_ans_0000155-3__dup3`, `medquad_ans_0000033-1__dup4`, `medquad_ans_0000058-11__dup1`
**Unsupported citations**: _none_

## Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Top docs**:

1. `medquad_ans_0000036_2-2` (score=0.6812) - Prostate cancer is most common in older men. In the U.S., about one out of five men will be diagnosed with prostate cancer. Most men diagnosed with prostate cancer do not die of it. See the following PDQ summaries for more information ab...
2. `medquad_ans_0000058-23` (score=0.6809) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But doctors also conside...
3. `medquad_ans_0000058-11` (score=0.6661) - Prostate cancer is one of the most common types of cancer among American men. It is a slow-growing disease that mostly affects older men. In fact, more than 60 percent of all prostate cancers are found in men over the age of 65. The dise...
4. `medquad_ans_0000036_1-4` (score=0.6569) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer (level of PSA, Gleason score, grade of the tumor, ho...
5. `medquad_ans_0000836-2` (score=0.6510) - About 1 in 7 men will be diagnosed with prostate cancer at some time during their life. In addition, studies indicate that many older men have undiagnosed prostate cancer that is non-aggressive and unlikely to cause symptoms or affect th...

**Answer**:

The treatments for prostate cancer may include watchful waiting, surgery, radiation therapy, or hormonal therapy. Some men receive a combination of these therapies. The choice of treatment depends on the stage of the disease and the grade of the tumor, as well as the man's age, general health, feelings about treatment and its possible side effects, and other health problems [medquad_ans_0000058-23]. A cure is probable for men whose prostate cancer is diagnosed early [medquad_ans_0000058-23].

**Citations detected**: `medquad_ans_0000058-23`
**Supported citations**: `medquad_ans_0000058-23`
**Unsupported citations**: _none_

## Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Top docs**:

1. `medquad_ans_0000196-1__dup3` (score=0.7648) - We all want high-quality health care, but it's hard to know how to choose. There are many things to consider, including - What your insurance covers - Whether a health care provider or service is accredited - The location of a service - ...
2. `medquad_ans_0000431-1__dup1` (score=0.7623) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when you need medical care....
3. `medquad_ans_0000869-1__dup1` (score=0.7530) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can bring: - Lists of your c...
4. `medquad_ans_0000362-1__dup2` (score=0.7416) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance might be available. Cer...
5. `medquad_ans_0000586-1__dup2` (score=0.7404) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state has its own rules. You...

**Answer**:

Yes, there is information about health insurance. Health insurance is a contract between you and your insurance company where you buy a plan, and the company agrees to pay part of your medical expenses when you need care. Many people in the United States get health insurance through their employers, and employers often help pay for it. Employer-based plans are often managed care plans that contract with healthcare providers to offer care at reduced costs. Individuals can also purchase health insurance on their own, and certain groups may qualify for government health insurance such as Medicare and Medicaid. The Affordable Care Act has expanded health insurance coverage for many people in the U.S. [medquad_ans_0000431-1__dup1]. Medicaid is a government health insurance program that assists low-income individuals in paying their medical bills, with federal guidelines and state-specific rules [medquad_ans_0000586-1__dup2]. Financial assistance may also be available

**Citations detected**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000586-1__dup2`
**Supported citations**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

## Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.6961) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...
2. `medquad_ans_0000635-3` (score=0.6952) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
3. `medquad_ans_0000635-2` (score=0.6845) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of all cases of intell...
4. `medquad_ans_0003860-5` (score=0.6842) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the diagnosis.
5. `medquad_ans_0003860-4` (score=0.6779) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome, one of the two sex...

**Answer**:

MECP2 duplication syndrome is a genetic disorder caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome [medquad_ans_0003860-3]. The MECP2 gene encodes a protein called MeCP2 that is critical for normal brain function, primarily by regulating the expression of other genes in the brain [medquad_ans_0000635-3]. An extra copy of the MECP2 gene leads to excess MeCP2 protein, which disrupts the proper regulation of gene expression, causing abnormal neuronal function and irregular brain activity, which in turn leads to the signs and symptoms of the syndrome [medquad_ans_0003860-3][medquad_ans_0000635-3]. The size of the duplication varies

**Citations detected**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Supported citations**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Unsupported citations**: _none_

## Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Top docs**:

1. `medquad_ans_0000063-6__dup1` (score=0.7124) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the cancer, whether or no...
2. `medquad_ans_0000063-29` (score=0.6954) - Yes. Some skin cancer patients take part in studies of new treatments. These studies, called clinical trials, are designed to find out whether a new treatment is both safe and effective. Clinical trials are research studies with people t...
3. `medquad_ans_0000028_2-7` (score=0.6873) - Key Points - There are different types of treatment for patients with melanoma. - Five types of standard treatment are used: - Surgery - Chemotherapy - Radiation therapy - Immunotherapy - Targeted therapy - New types of treatment are bei...
4. `medquad_ans_0000028_2-8` (score=0.6843) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website Treatment for melanoma may cause side effects. For information about side effects caused by treatment for can...
5. `medquad_ans_0000063-3__dup3` (score=0.6805) - Early Detection is Important When skin cancer is found early, it is more likely to be treated successfully. Therefore, it is important to know how to recognize the signs of skin cancer in order to improve the chances of early diagnosis. ...

**Answer**:

The provided context does not contain information about the outlook for melanoma. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000063-6__dup1]

**Citations detected**: `medquad_ans_0000063-6__dup1`
**Supported citations**: `medquad_ans_0000063-6__dup1`
**Unsupported citations**: _none_

## Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Top docs**:

1. `medquad_ans_0000132-2` (score=0.7430) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
2. `medquad_ans_0000132-5` (score=0.6829) - These resources address the diagnosis or management of boomerang dysplasia: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Boomerang dysplasia These resources from MedlinePlus offer information about the d...
3. `medquad_ans_0000438-2` (score=0.6806) - Greenberg dysplasia is a very rare condition. Approximately ten cases have been reported in the scientific literature.
4. `medquad_ans_0006111-4` (score=0.6742) - Is Tietze syndrome inherited? Tietze syndrome is not thought to be inherited. Most cases occur sporadically in people with no family history of the condition.
5. `medquad_ans_0006133-4` (score=0.6696) - Is Townes-Brocks syndrome genetic? Yes. Townes-Brocks syndrome is inherited in an autosomal dominant fashion, which means that one copy of the altered gene in each cell is sufficient to cause the disorder. In about 50% of cases, an affec...

**Answer**:

The provided context does not contain information on whether boomerang dysplasia is inherited. [medquad_ans_0000132-2] states that boomerang dysplasia is a rare disorder with an unknown prevalence and mentions that approximately 10 affected individuals have been identified, but it does not address inheritance. [medquad_ans_0000132-5] lists resources for diagnosis and management, including genetic counseling, but does not specify inheritance patterns. Therefore, based on the given context, the inheritance status of boomerang dysplasia cannot be determined.

**Citations detected**: `medquad_ans_0000132-2`, `medquad_ans_0000132-5`
**Supported citations**: `medquad_ans_0000132-2`, `medquad_ans_0000132-5`
**Unsupported citations**: _none_

## Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Top docs**:

1. `medquad_ans_0000466-5` (score=0.6694) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - Genetic Testing Regis...
2. `medquad_ans_0002838-3` (score=0.6666) - How is hereditary multiple osteochondromas inherited? HMO is caused by mutations in the EXT1 and EXT2 genes. It is inherited in an autosomal dominant pattern, which means that one copy of the altered gene in each cell is sufficient to ca...
3. `medquad_ans_0002838-1` (score=0.6549) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called osteochondromas). T...
4. `medquad_ans_0000466-3` (score=0.6531) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exostosin proteins bind to...
5. `medquad_ans_0002838-4` (score=0.6505) - Is genetic testing available for hereditary multiple osteochondromas? GeneTests lists the names of laboratories that are performing genetic testing for hereditary multiple osteochondromas. To view the contact information for the clinical...

**Answer**:

Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas [medquad_ans_0002838-3, medquad_ans_0002838-1]. These mutations lead to the production of abnormal proteins, exostosin-1 and exostosin-2, which are involved in the formation of bone and cartilage [medquad_ans_0000466-3]. The condition is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0002838-3].

**Citations detected**: `medquad_ans_0002838-3, medquad_ans_0002838-1`, `medquad_ans_0000466-3`, `medquad_ans_0002838-3`
**Supported citations**: `medquad_ans_0000466-3`, `medquad_ans_0002838-3`
**Unsupported citations**: `medquad_ans_0002838-3, medquad_ans_0002838-1`

## Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Top docs**:

1. `medquad_ans_0000485-2` (score=0.7119) - Holt-Oram syndrome is estimated to affect 1 in 100,000 individuals.
2. `medquad_ans_0000738-2` (score=0.7065) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
3. `medquad_ans_0001036-2` (score=0.6996) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
4. `medquad_ans_0000746-1` (score=0.6987) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abnormalities associate...
5. `medquad_ans_0000135-2` (score=0.6984) - Branchio-oculo-facial syndrome is a rare condition, although the prevalence is unknown.

**Answer**:

The provided context does not specify the exact number of people affected by oculofaciocardiodental (OFCD) syndrome. It states that OFCD syndrome occurs only in females and describes its features, but the prevalence or affected population size is not given [medquad_ans_0000746-1]. Therefore, based on the available information, the number of people affected by OFCD syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000746-1`
**Supported citations**: `medquad_ans_0000746-1`
**Unsupported citations**: _none_

## Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Top docs**:

1. `medquad_ans_0000189-2` (score=0.7040) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100,000 to 230,000 indi...
2. `medquad_ans_0001350-4` (score=0.6988) - How is citrullinemia type I inherited? Citrullinemia type I is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive conditi...
3. `medquad_ans_0001350-5` (score=0.6817) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention, hyperammonemia and...
4. `medquad_ans_0000445-3__dup1` (score=0.6755) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). This may be accomplis...
5. `medquad_ans_0001350-1` (score=0.6662) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases called urea cycle disord...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood due to a deficiency of the enzyme arginase. Type I citrullinemia is the most common form, affecting about 1 in 57,000 people worldwide, and is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000189-2][medquad_ans_0001350-4]. Type II citrullinemia is found primarily in the Japanese population, occurring in an estimated 1 in 100,000 to 230,000 individuals, and has also been reported in other populations, including East Asia and the Middle East [medquad_ans_0000189-2]. Untreated severe cases of Type I citr

**Citations detected**: `medquad_ans_0000189-2`, `medquad_ans_0001350-4`
**Supported citations**: `medquad_ans_0000189-2`, `medquad_ans_0001350-4`
**Unsupported citations**: _none_

## Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000807-1` (score=0.6642) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops too many folds, and...
2. `medquad_ans_0000720-1__dup1` (score=0.6589) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the entire surface of the br...
3. `medquad_ans_0004956-1` (score=0.6554) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associated with the condition...
4. `medquad_ans_0000722-1__dup1` (score=0.6475) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of the face, tongue, jaws...
5. `medquad_ans_0000807-2` (score=0.6438) - The prevalence of isolated polymicrogyria is unknown. Researchers believe that it may be relatively common overall, although the individual forms of the disorder (such as bilateral generalized polymicrogyria) are probably rare.

**Answer**:

The provided context does not mention "Bilateral frontal polymicrogyria" or any form of polymicrogyria specifically affecting the frontal lobe. Therefore, based on the available information, it is not possible to answer the question. [medquad_ans_0000807-1] and [medquad_ans_0000720-1__dup1] discuss bilateral generalized polymicrogyria and other forms of polymicrogyria but do not specify bilateral frontal polymicrogyria.

**Citations detected**: `medquad_ans_0000807-1`, `medquad_ans_0000720-1__dup1`
**Supported citations**: `medquad_ans_0000807-1`, `medquad_ans_0000720-1__dup1`
**Unsupported citations**: _none_

## Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Top docs**:

1. `medquad_ans_0000944-2` (score=0.6676) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0000949-5` (score=0.6674) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from MedlinePlus offer informa...
3. `medquad_ans_0000943-5` (score=0.6610) - These resources address the diagnosis or management of spondylocarpotarsal synostosis syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Spondylocarpotarsal synostosis syndrome These resources from M...
4. `medquad_ans_0000943-2` (score=0.6592) - Spondylocarpotarsal synostosis syndrome is a rare disorder; its prevalence is unknown. At least 25 affected individuals have been identified.
5. `medquad_ans_0000944-5` (score=0.6538) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic Testing Registry: Sp...

**Answer**:

The provided context does not specify the treatments for spondylocostal dysostosis. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, surgery and rehabilitation, and palliative care, it does not detail specific treatments for spondylocostal dysostosis [medquad_ans_0000944-5]. Therefore, based on the given information, the exact treatments cannot be determined.

**Citations detected**: `medquad_ans_0000944-5`
**Supported citations**: `medquad_ans_0000944-5`
**Unsupported citations**: _none_

## Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Top docs**:

1. `medquad_ans_0000786-2` (score=0.7044) - Perry syndrome is very rare; about 50 affected individuals have been reported worldwide.
2. `medquad_ans_0000786-1` (score=0.6436) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (hypoventilation). The...
3. `medquad_ans_0000785-2` (score=0.6387) - Perrault syndrome is a rare disorder; fewer than 100 affected individuals have been described in the medical literature. It is likely that the condition is underdiagnosed, because males without an affected sister will likely be misdiagno...
4. `medquad_ans_0000613-2` (score=0.6382) - Mabry syndrome is likely a rare condition, but its prevalence is unknown. More than 20 cases have been described in the scientific literature.
5. `medquad_ans_0000145-3__dup3` (score=0.6320) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breath and coughing, some...

**Answer**:

Perry syndrome is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (hypoventilation). Signs of parkinsonism include unusually slow movements (bradykinesia), stiffness, and tremors. These movement abnormalities are often accompanied by changes in personality and behavior. The most frequent psychiatric changes include depression, a general loss of interest and enthusiasm (apathy), withdrawal from friends and family, and suicidal thoughts. Many affected individuals also experience significant, unexplained weight loss early in the disease. Hypoventilation is a later feature, most often occurring at night, causing affected individuals to wake up frequently. As the disease worsens, hypoventilation can result in a life-threatening lack of oxygen and respiratory failure [medquad_ans_0000786-1].

**Citations detected**: `medquad_ans_0000786-1`
**Supported citations**: `medquad_ans_0000786-1`
**Unsupported citations**: _none_

## Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Top docs**:

1. `medquad_ans_0001028-2` (score=0.7097) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people in the U.S. who are...
2. `medquad_ans_0001027-2` (score=0.7062) - Warfarin resistance is thought to be a rare condition, although its prevalence is unknown.
3. `medquad_ans_0001028-1` (score=0.6891) - Warfarin sensitivity is a condition in which individuals have a low tolerance for the drug warfarin. Warfarin is an anticoagulant, which means that it thins the blood, preventing blood clots from forming. Warfarin is often prescribed to ...
4. `medquad_ans_0001028-5` (score=0.6815) - These resources address the diagnosis or management of warfarin sensitivity: - Food and Drug Administration Medication Guide - MedlinePlus Drugs & Supplements: Warfarin - My46 Trait Profile - PharmGKB - WarfarinDosing.org These resources...
5. `medquad_ans_0001027-5` (score=0.6764) - These resources address the diagnosis or management of warfarin resistance: - American Society of Hematology: Antithrombotic Therapy - MedlinePlus Drugs & Supplements: Warfarin - PharmGKB These resources from MedlinePlus offer informatio...

**Answer**:

The most common sign of warfarin sensitivity is excessive internal bleeding, which often occurs when individuals with warfarin sensitivity are given too much of the medication [medquad_ans_0001028-2]. This excessive bleeding can affect the brain, gastrointestinal tract, or other tissues and may lead to serious health problems or death if not managed properly [medquad_ans_0001028-1].

**Citations detected**: `medquad_ans_0001028-2`, `medquad_ans_0001028-1`
**Supported citations**: `medquad_ans_0001028-2`, `medquad_ans_0001028-1`
**Unsupported citations**: _none_

## Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Top docs**:

1. `medquad_ans_0000701-2` (score=0.7681) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
2. `medquad_ans_0004262-3` (score=0.7643) - What causes Myhre syndrome? Myhre syndrome is caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus. This signaling path...
3. `medquad_ans_0000701-3` (score=0.7474) - Mutations in the SMAD4 gene cause Myhre syndrome. The SMAD4 gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus. This signaling pathway, called the transforming g...
4. `medquad_ans_0004262-4` (score=0.7322) - How is Myhre syndrome inherited? This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new (de novo) mutations in t...
5. `medquad_ans_0000700-2` (score=0.7219) - The incidence of MYH9-related disorder is unknown. More than 200 affected families have been reported in the scientific literature.

**Answer**:

Myhre syndrome is a rare disorder caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus through the transforming growth factor beta (TGF-) pathway. The SMAD4 protein interacts with other proteins to control the activity of particular genes that influence many areas of development. Mutations in the SMAD4 gene may impair the protein's ability to bind properly with other proteins in the signaling pathway or result in an abnormally stable SMAD4 protein that remains active longer. These changes lead to abnormal signaling in many cell types, affecting the development of several body systems and causing the signs and symptoms of Myhre syndrome. The disorder is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder, although most cases result from new (de novo) mutations. Only about 30 cases have been documented

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Top docs**:

1. `medquad_ans_0000976-2` (score=0.6978) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.
2. `medquad_ans_0006057-4` (score=0.6905) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of thanatophoric dysplasia...
3. `medquad_ans_0000976-4` (score=0.6852) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are caused by new mutations...
4. `medquad_ans_0006057-3` (score=0.6695) - What causes thanatophoric dysplasia? Thanatophoric dysplasia is caused by mutations in the FGFR3 gene. This gene provides instructions for making a protein that is involved in the development and maintenance of bone and brain tissue. Mut...
5. `medquad_ans_0000976-5` (score=0.6688) - These resources address the diagnosis or management of thanatophoric dysplasia: - Gene Review: Gene Review: Thanatophoric Dysplasia - Genetic Testing Registry: Thanatophoric dysplasia type 1 - Genetic Testing Registry: Thanatophoric dysp...

**Answer**:

The provided context does not include information about the symptoms of thanatophoric dysplasia. Therefore, a specific answer cannot be given based on the available content. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Top docs**:

1. `medquad_ans_0000381-2` (score=0.7185) - Fragile X syndrome occurs in approximately 1 in 4,000 males and 1 in 8,000 females.
2. `medquad_ans_0002417-1` (score=0.6956) - Fragile X syndrome is a genetic condition involving changes in part of the X chromosome. This condition causes a range of developmental problems including learning disabilities and cognitive impairment. It is the most common form of inhe...
3. `medquad_ans_0000381-5` (score=0.6953) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Genetic Testing Regist...
4. `medquad_ans_0000381-3` (score=0.6877) - Mutations in the FMR1 gene cause fragile X syndrome. The FMR1 gene provides instructions for making a protein called FMRP. This protein helps regulate the production of other proteins and plays a role in the development of synapses, whic...
5. `medquad_ans_0000381-4` (score=0.6855) - Fragile X syndrome is inherited in an X-linked dominant pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. (The Y chromosome is the oth...

**Answer**:

Symptoms of Fragile X syndrome include learning disabilities, cognitive impairment, symptoms of autism spectrum disorders, seizures, and characteristic physical features. [medquad_ans_0002417-1]

**Citations detected**: `medquad_ans_0002417-1`
**Supported citations**: `medquad_ans_0002417-1`
**Unsupported citations**: _none_

## Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.6779) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000056-3__dup1` (score=0.6490) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.
3. `medquad_ans_0000889-4__dup1` (score=0.6451) - How is CADASIL inherited? CADASIL is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause CADASIL. In most cases, an affected person inherit...
4. `medquad_ans_0000166-1` (score=0.6444) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow in small blood vess...
5. `medquad_ans_0000889-3__dup1` (score=0.6383) - What causes CADASIL? CADASIL is caused by a mutation in the NOTCH3 gene. The NOTCH3 gene gives the body instructions to make the Notch3 receptor protein, needed for normal function and survival of vascular smooth muscle cells. Mutations ...

**Answer**:

CADASIL, or cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, is an inherited condition that causes stroke and other impairments by affecting blood flow in small blood vessels, particularly in the brain. It results from abnormal vascular smooth muscle cells surrounding these blood vessels, which gradually die, leading to arteriopathy. This damage can cause migraines (often with visual sensations or auras), recurrent seizures (epilepsy), and areas of tissue death (infarcts), especially in the subcortical regions of the brain, which are involved in reasoning and memory. These infarcts can lead to strokes that may occur at any time from childhood to late adulthood, typically during mid-adulthood, and often recur throughout a person's lifetime. Recurrent strokes can result in progressive cognitive decline, dementia, mood and personality changes, and leukoencephalopathy—a change in brain

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Top docs**:

1. `medquad_ans_0000688-2` (score=0.7097) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followed by FMTC. Type 2B i...
2. `medquad_ans_0000688-1` (score=0.7042) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cel...
3. `medquad_ans_0000688-5` (score=0.7002) - These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Genetic Testing Regist...
4. `medquad_ans_0004185-3` (score=0.6906) - What causes multiple endocrine neoplasia, type 1? Multiple endocrine neoplasia, type 1 (MEN1) is caused by mutations in the MEN1 gene. MEN1 is a tumor suppressor gene which means that it encodes a protein that helps keep cells from growi...
5. `medquad_ans_0004185-6` (score=0.6882) - How might multiple endocrine neoplasia, type 1 be treated? People with multiple endocrine neoplasia, type 1 (MEN1) are usually managed with regular screening to allow for early diagnosis and treatment of endocrine tumors. This screening ...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with multiple endocrine neoplasia. Type 1 frequently involves tumors of the

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Top docs**:

1. `medquad_ans_0004155-4` (score=0.7238) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
2. `medquad_ans_0004155-1` (score=0.7177) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by mutations in the IDUA...
3. `medquad_ans_0000680-2` (score=0.7140) - MPS III is the most common type of mucopolysaccharidosis; the estimated incidence of all four types combined is 1 in 70,000 newborns. MPS IIIA and MPS IIIB are much more common than MPS IIIC and MPS IIID.
4. `medquad_ans_0004155-3` (score=0.7059) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs)....
5. `medquad_ans_0000683-2` (score=0.7023) - The exact incidence of MPS VII is unknown, although it is estimated to occur in 1 in 250,000 newborns. It is one of the rarest types of mucopolysaccharidosis.

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I. These mutations lead to reduced levels or the complete lack of the IDUA enzyme, which is responsible for breaking down large sugar molecules called glycosaminoglycans (GAGs). The lack of IDUA enzyme activity results in the accumulation of GAGs within lysosomes, causing lysosomal enlargement and affecting multiple organs and tissues [medquad_ans_0004155-1][medquad_ans_0004155-3].

**Citations detected**: `medquad_ans_0004155-1`, `medquad_ans_0004155-3`
**Supported citations**: `medquad_ans_0004155-1`, `medquad_ans_0004155-3`
**Unsupported citations**: _none_

## Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Top docs**:

1. `medquad_ans_0005065-1` (score=0.6627) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the pleural space (the space between the lungs and the chest cavity) that can result in the partial or complete collapse of a lung. It is called primary because it oc...
2. `medquad_ans_0000825-4` (score=0.6547) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, a person inherits th...
3. `medquad_ans_0000825-5` (score=0.6518) - These resources address the diagnosis or management of primary spontaneous pneumothorax: - Genetic Testing Registry: Pneumothorax, primary spontaneous - MedlinePlus Encyclopedia: Chest Tube Insertion - MedlinePlus Encyclopedia: Collapsed...
4. `medquad_ans_0000825-3` (score=0.6385) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called folliculin. In the lungs, f...
5. `medquad_ans_0005065-2` (score=0.6290) - What are the signs and symptoms of Primary spontaneous pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Primary spontaneous pneumothorax. If the information is available, the table below in...

**Answer**:

Genetic changes related to primary spontaneous pneumothorax include mutations in the FLCN gene, which can cause the condition and follow an autosomal dominant pattern of inheritance [medquad_ans_0005065-1]. In these cases, one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0000825-4]. It is estimated that only 40 percent of individuals with an FLCN gene mutation go on to develop primary spontaneous pneumothorax [medquad_ans_0000825-4]. Additionally, several genetic disorders have been linked to primary spontaneous pneumothorax, including Marfan syndrome, homocystinuria, and Birt-Hogg-Dube syndrome [medquad_ans_0005065-1].

**Citations detected**: `medquad_ans_0005065-1`, `medquad_ans_0000825-4`
**Supported citations**: `medquad_ans_0005065-1`, `medquad_ans_0000825-4`
**Unsupported citations**: _none_

## Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Top docs**:

1. `medquad_ans_0000066-26` (score=0.7252) - With stroke, treatment depends on the stage of the disease. There are three treatment stages for stroke: prevention, therapy immediately after stroke, and rehabilitation after stroke. Stroke treatments include medications, surgery, and r...
2. `medquad_ans_0000066-13` (score=0.7167) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blood vessel in the bra...
3. `medquad_ans_0000066-1__dup4` (score=0.7052) - Stroke -- A Serious Event A stroke is serious, just like a heart attack. Each year in the United States, approximately 795,000 people have a stroke. About 610,000 of these are first or new strokes. On average, one American dies from stro...
4. `medquad_ans_0000066-19` (score=0.7027) - A risk factor is a condition or behavior that increases your chances of getting a disease. Having a risk factor for stroke doesn't mean you'll have a stroke. On the other hand, not having a risk factor doesn't mean you'll avoid a stroke....
5. `medquad_ans_0000066-12` (score=0.7026) - Stroke occurs in all age groups, in both sexes, and in all races in every country. It can even occur before birth, when the fetus is still in the womb. Studies show the risk of stroke doubles for each decade between the ages of 55 and 85...

**Answer**:

A stroke is sometimes called a "brain attack." Most often, stroke occurs when blood flow to the brain stops because it is blocked by a clot. When this happens, the brain cells in the immediate area begin to die. Some brain cells die because they stop getting the oxygen and nutrients they need to function. Other brain cells die because they are damaged by sudden bleeding into or around the brain. The brain cells that don't die immediately remain at risk for death. These cells can linger in a compromised or weakened state for several hours. With timely treatment, these cells can be saved [medquad_ans_0000066-1__dup4].

**Citations detected**: `medquad_ans_0000066-1__dup4`
**Supported citations**: `medquad_ans_0000066-1__dup4`
**Unsupported citations**: _none_

## Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Top docs**:

1. `medquad_ans_0006276-5` (score=0.7385) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. When the individual ...
2. `medquad_ans_0006276-4` (score=0.7226) - Is uncombable hair syndrome inherited? Uncombable hair syndrome (UHS) is thought to be inherited in an autosomal dominant manner with reduced penetrance. Autosomal dominant means that having a change (mutation) in only one copy of the re...
3. `medquad_ans_0006276-2` (score=0.7084) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of hair remains normal...
4. `medquad_ans_0006276-3` (score=0.7013) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been suggested that the co...
5. `medquad_ans_0006276-1` (score=0.6939) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed flat. It may first b...

**Answer**:

The provided context does not specify the exact cause of uncombable hair syndrome (UHS). It mentions that UHS is thought to be inherited in an autosomal dominant manner with reduced penetrance and that a mutation in the responsible gene is sufficient to cause features of the condition [medquad_ans_0006276-4]. However, no specific gene or molecular cause is identified in the given text. Therefore, based on the available information, the cause of UHS cannot be determined.

**Citations detected**: `medquad_ans_0006276-4`
**Supported citations**: `medquad_ans_0006276-4`
**Unsupported citations**: _none_

## Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Top docs**:

1. `medquad_ans_0005440-4` (score=0.7343) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is inherited. In some ...
2. `medquad_ans_0000887-2` (score=0.7336) - The exact incidence of Russell-Silver syndrome is unknown, but the condition is estimated to affect 1 in 75,000 to 100,000 people.
3. `medquad_ans_0000887-4` (score=0.7264) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected families, the condition ap...
4. `medquad_ans_0000887-5` (score=0.7109) - These resources address the diagnosis or management of Russell-Silver syndrome: - Gene Review: Gene Review: Russell-Silver Syndrome - Genetic Testing Registry: Russell-Silver syndrome - MedlinePlus Encyclopedia: Russell-Silver syndrome T...
5. `medquad_ans_0005440-3` (score=0.7104) - What causes Russell-Silver syndrome? Russell-Silver syndrome (RSS) is a genetic disorder that usually results from the abnormal regulation of certain genes that control growth. Two genetic causes have been found to result in the majority...

**Answer**:

Russell-Silver syndrome is not always inherited; most cases are sporadic, meaning they occur in people with no family history of the disorder. However, less commonly, it can be inherited in either an autosomal dominant or autosomal recessive manner. In autosomal dominant inheritance, one copy of a genetic change in each cell is sufficient to cause the disorder. In autosomal recessive inheritance, both copies of the responsible gene must be altered for the disorder to manifest, and the parents are typically carriers without symptoms [medquad_ans_0005440-4]. Therefore, while inheritance is possible in some families, it is not the rule. [medquad_ans_0004887-2]

**Citations detected**: `medquad_ans_0005440-4`, `medquad_ans_0004887-2`
**Supported citations**: `medquad_ans_0005440-4`
**Unsupported citations**: `medquad_ans_0004887-2`

## Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.7381) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weigh...
2. `medquad_ans_0000095-7` (score=0.7125) - Successful weight-loss treatments include setting goals and making lifestyle changes, such as eating fewer calories and being physically active. Medicines and weight-loss surgery also are options for some people if lifestyle changes aren...
3. `medquad_ans_0000022-11` (score=0.6991) - Your eating, diet, and nutrition choices play an important role in preventing or delaying diabetes. Follow the suggestions below to reach and maintain a reasonable weight and make wise food choices most of the time. Remember that it can ...
4. `medquad_ans_0000095-1__dup2` (score=0.6936) - Espaol The terms "overweight" and "obesity" refer to body weight thats greater than what is considered healthy for a certain height. The most useful measure of overweight and obesity is body mass index (BMI). BMI is calculated from your ...
5. `medquad_ans_0000036-27` (score=0.6931) - Blood pressure rises as body weight increases. Losing even 10 pounds can lower blood pressure -- and it has the greatest effect for those who are overweight and already have hypertension. If you are overweight or obese, work with your he...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Citations detected**: `medquad_ans_0000969-1__dup2`
**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

## Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Top docs**:

1. `medquad_ans_0000889-2` (score=0.7115) - Saethre-Chotzen syndrome has an estimated prevalence of 1 in 25,000 to 50,000 people.
2. `medquad_ans_0000889-5` (score=0.6583) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Saethre-Chotzen syndrom...
3. `medquad_ans_0000889-4` (score=0.6478) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one affected parent. Ot...
4. `medquad_ans_0000889-3` (score=0.6475) - Mutations in the TWIST1 gene cause Saethre-Chotzen syndrome. The TWIST1 gene provides instructions for making a protein that plays an important role in early development. This protein is a transcription factor, which means that it attach...
5. `medquad_ans_0006111-6` (score=0.6345) - How might Tietze syndrome be treated? In some individuals, the pain associated with Tietze syndrome resolves on its own without any treatment. Management options for others may include avoidance of strenuous activity; applying local heat...

**Answer**:

The provided context does not specify the treatments for Saethre-Chotzen syndrome. While it mentions resources that address diagnosis or management, such as genetic counseling, surgical intervention, and rehabilitation, it does not detail specific treatment options. Therefore, based on the given information, the treatments for Saethre-Chotzen syndrome cannot be determined. [medquad_ans_0000889-5]

**Citations detected**: `medquad_ans_0000889-5`
**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

## Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000512-2` (score=0.6502) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Familial pulmonary fibros...
2. `medquad_ans_0000758-1__dup2` (score=0.6359) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...
3. `medquad_ans_0000512-4` (score=0.6351) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance. Autosomal dominant ...
4. `medquad_ans_0000078-6` (score=0.6302) - Idiopathic pulmonary fibrosis (IPF) causes the same kind of scarring and symptoms as some other lung diseases. This makes it hard to diagnose. Seeking medical help as soon as you have symptoms is important. If possible, seek care from a ...
5. `medquad_ans_0000512-5` (score=0.6258) - These resources address the diagnosis or management of idiopathic pulmonary fibrosis: - Gene Review: Gene Review: Pulmonary Fibrosis, Familial - Genetic Testing Registry: Idiopathic fibrosing alveolitis, chronic form These resources from...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a condition in which the tissue deep in the lungs becomes scarred over time, causing the tissue to thicken and stiffen. This scarring makes it difficult to catch one's breath and can result in insufficient oxygen in the blood. The exact cause of IPF is often unknown, and this is referred to as idiopathic pulmonary fibrosis. Most cases are sporadic, occurring in individuals with no family history of the disorder, while a small percentage appear to run in families and follow an autosomal dominant pattern of inheritance with reduced penetrance. Symptoms include shortness of breath, a dry hacking cough, fatigue, weight loss, muscle and joint aches, and clubbing of the fingers or toes. Diagnosis is based on medical history, physical exam, imaging tests, and lung function tests, and there is currently no cure; treatment focuses on managing symptoms and improving quality of life [medquad*ans*

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Top docs**:

1. `medquad_ans_0000213-4__dup1` (score=0.7645) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
2. `medquad_ans_0000213-3__dup1` (score=0.7526) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recovers, although there i...
3. `medquad_ans_0000207-4__dup1` (score=0.7159) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research relating to myoclonus in its laboratories at the National Institutes of Health (NIH) and also supports additional research through grants to major medi...
4. `medquad_ans_0000213-1__dup1` (score=0.7079) - Neuroleptic malignant syndrome is a life-threatening, neurological disorder most often caused by an adverse reaction to neuroleptic or antipsychotic drugs. Symptoms include high fever, sweating, unstable blood pressure, stupor, muscular ...
5. `medquad_ans_0000264-4__dup1` (score=0.7076) - The NINDS conducts and supports a broad range of research on movement disorders including tardive dyskinesia. The goals of this research are to improve understanding of these disorders and to discover ways to treat, prevent, and, ultimat...

**Answer**:

Research on neuroleptic malignant syndrome is being conducted by the National Institute of Neurological Disorders and Stroke (NINDS), which supports research focused on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. Early identification and treatment of individuals with the syndrome improve outcomes [medquad_ans_0000213-3__dup1]. Additionally, when clinically indicated, a low potency neuroleptic can be reintroduced very slowly after recovery, though there is a risk of recurrence; alternatively, another class of drugs may be used to substitute for the neuroleptic [medquad_ans_0000213-3__dup1]. Anesthesia may pose a risk to individuals who have experienced the syndrome [medquad_ans_0000213-3__dup1]. The NINDS also supports broader research on neurological disorders, including

**Citations detected**: `medquad_ans_0000213-4__dup1`, `medquad_ans_0000213-3__dup1`
**Supported citations**: `medquad_ans_0000213-4__dup1`, `medquad_ans_0000213-3__dup1`
**Unsupported citations**: _none_

## Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Top docs**:

1. `medquad_ans_0000203-7` (score=0.6942) - People with kidney stones may have pain while urinating, see blood in the urine, or feel a sharp pain in the back or lower abdomen. The pain may last for a short or long time. People may experience nausea and vomiting with the pain. Howe...
2. `medquad_ans_0000203-11` (score=0.6866) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary tract. - Certain fo...
3. `medquad_ans_0000204-9` (score=0.6844) - To prevent kidney stones, health care providers and their patients must understand what is causing the stones to form. Especially in children with suspected metabolic abnormalities or with recurrent stones, a 24-hour urine collection is ...
4. `medquad_ans_0000221-8` (score=0.6844) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other health problems. The d...
5. `medquad_ans_0000221-9` (score=0.6804) - The treatment for kidney stones usually depends on their size and what they are made of. Kidney stones may be treated by your regular doctor or by a urologista doctor who specializes in the urinary tract. You may need treatment if you ha...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, including questions about family history of kidney stones, diet, digestive problems, and other health issues. The provider may perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can show whether there is an infection or if the urine contains substances that form stones. Blood tests can show problems such as abnormal levels of minerals or electrolytes that may contribute to stone formation [medquad_ans_0000203-11][medquad_ans_0000221-8].

**Citations detected**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`
**Supported citations**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`
**Unsupported citations**: _none_

## Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.6863) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of bladder cancer can a...
2. `medquad_ans_0000041_1-6` (score=0.6620) - Key Points - There are different types of treatment for patients with urethral cancer. - Four types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Active surveillance - New types of treatment are being tes...
3. `medquad_ans_0000041_1-4` (score=0.6541) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - Where the cancer formed in the urethra. - Whether the cancer has spread thro...
4. `medquad_ans_0000041_1-5` (score=0.6463) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. - Cancer may spread f...
5. `medquad_ans_0000128-2` (score=0.6271) - In the United States, bladder cancer is the fourth most common type of cancer in men and the ninth most common cancer in women. About 45,000 men and 17,000 women are diagnosed with bladder cancer each year.

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Top docs**:

1. `medquad_ans_0000771-2` (score=0.7066) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.
2. `medquad_ans_0000709-2` (score=0.6870) - Myotonia congenita is estimated to affect 1 in 100,000 people worldwide. This condition is more common in northern Scandinavia, where it occurs in approximately 1 in 10,000 people.
3. `medquad_ans_0000771-5` (score=0.6693) - These resources address the diagnosis or management of paramyotonia congenita: - Genetic Testing Registry: Paramyotonia congenita of von Eulenburg - Periodic Paralysis International: How is Periodic Paralysis Diagnosed? These resources f...
4. `medquad_ans_0000701-2` (score=0.6630) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
5. `medquad_ans_0000700-2` (score=0.6598) - The incidence of MYH9-related disorder is unknown. More than 200 affected families have been reported in the scientific literature.

**Answer**:

The provided context does not specify whether paramyotonia congenita is inherited. While it mentions that paramyotonia congenita is an uncommon disorder affecting fewer than 1 in 100,000 people and references genetic testing resources, there is no direct information on its inheritance pattern [medquad_ans_0000771-2][medquad_ans_0000771-5]. Therefore, based on the given context, the inheritance status of paramyotonia congenita cannot be determined.

**Citations detected**: `medquad_ans_0000771-2`, `medquad_ans_0000771-5`
**Supported citations**: `medquad_ans_0000771-2`, `medquad_ans_0000771-5`
**Unsupported citations**: _none_

## Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Top docs**:

1. `medquad_ans_0000125-2` (score=0.7498) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
2. `medquad_ans_0000125-4` (score=0.7229) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the mutated gene, but the...
3. `medquad_ans_0000125-5` (score=0.7080) - These resources address the diagnosis or management of biotinidase deficiency: - Baby's First Test - Gene Review: Gene Review: Biotinidase Deficiency - Genetic Testing Registry: Biotinidase deficiency - MedlinePlus Encyclopedia: Pantothe...
4. `medquad_ans_0000125-1` (score=0.7007) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few months of life, alth...
5. `medquad_ans_0000735-1__dup1` (score=0.6913) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form of the disorder is c...

**Answer**:

The symptoms of biotinidase deficiency include delayed development, seizures, weak muscle tone (hypotonia), breathing problems, hearing and vision loss, problems with movement and balance (ataxia), skin rashes, hair loss (alopecia), and a fungal infection called candidiasis in the more severe form, profound biotinidase deficiency. In partial biotinidase deficiency, without treatment, affected children may experience hypotonia, skin rashes, and hair loss, which may only appear during illness, infection, or other times of stress [medquad_ans_0000125-1].

**Citations detected**: `medquad_ans_0000125-1`
**Supported citations**: `medquad_ans_0000125-1`
**Unsupported citations**: _none_

## Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Top docs**:

1. `medquad_ans_0000298-1__dup1` (score=0.7504) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includes rivers, lakes and...
2. `medquad_ans_0000331-1__dup1` (score=0.6997) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in the home can trigger a...
3. `medquad_ans_0000422-1__dup1` (score=0.6954) - Summary : Even if you use them properly, many chemicals can still harm human health and the environment. When you throw these substances away, they become hazardous waste. Some hazardous wastes come from products in our homes. Our garbag...
4. `medquad_ans_0000055-1__dup3` (score=0.6891) - Summary : Arsenic is a natural element found in soil and minerals. Arsenic compounds are used to preserve wood, as pesticides, and in some industries. Arsenic can get into air, water, and the ground from wind-blown dust. It may also get ...
5. `medquad_ans_0000040-13` (score=0.6879) - For the most part, no one knows why some people develop leukemia and others do not. Most people who have known risk factors do not get leukemia, while many who get the disease do not have any risk factors. Studies have identified the fol...

**Answer**:

Yes, the provided context includes information about water pollution. It states that environmental pollution, including in water, can lead to health problems. Specifically, it mentions that pollutants in air or in the home can trigger asthma attacks, and that chemicals in the environment, such as those from industrial use or mining, can enter water systems. Additionally, arsenic can get into water from runoff and wind-blown dust, and exposure to arsenic through drinking water is a noted health concern [medquad_ans_0000331-1__dup1]. The context also notes that drinking water quality can vary based on the condition of the source water and treatment, and that water supplies may contain contaminants [medquad_ans_0000298-1__dup1].

Thus, water pollution is discussed in relation to contaminants like arsenic, industrial runoff, and the overall quality of drinking water [medquad_ans_0000

**Citations detected**: `medquad_ans_0000331-1__dup1`, `medquad_ans_0000298-1__dup1`
**Supported citations**: `medquad_ans_0000331-1__dup1`, `medquad_ans_0000298-1__dup1`
**Unsupported citations**: _none_

## Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.7254) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue ...
2. `medquad_ans_0000132-4__dup1` (score=0.6914) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and also supports additiona...
3. `medquad_ans_0000132-1__dup1` (score=0.6670) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend their knees to lift t...
4. `medquad_ans_0000067-25` (score=0.6660) - Talk to your doctor when you first notice symptoms of neuropathy. Certain medications and other approaches, such as physical therapy, may help alleviate symptoms. There are some steps you can take yourself. Pay careful attention to your ...
5. `medquad_ans_0000132-2__dup1` (score=0.6572) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the muscles and maintain jo...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Citations detected**: `medquad_ans_0000132-3__dup1`
**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

## Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Top docs**:

1. `medquad_ans_0000145-2` (score=0.7677) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
2. `medquad_ans_0000916-1__dup1` (score=0.7593) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
3. `medquad_ans_0000915-1__dup1` (score=0.7593) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
4. `medquad_ans_0000916-5__dup1` (score=0.7559) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene known to be associa...
5. `medquad_ans_0000916-4__dup1` (score=0.7532) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is enough to cause fea...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs, and in some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people develop pain or weakness by adolescence. The disease is caused by a mutation in the TGFB1 gene, which is inherited in an autosomal dominant fashion. In some instances, individuals have the gene mutation but never develop the characteristic features, or features are present without a detectable mutation, referred to as Camurati-Engelmann disease type II. Diagnosis is based on physical examination, radiographic findings, and can be confirmed by molecular genetic testing, with mutations in TGFB

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Top docs**:

1. `medquad_ans_0000316-2` (score=0.7027) - X-linked Emery-Dreifuss muscular dystrophy is the most common form of this condition, affecting an estimated 1 in 100,000 people. The autosomal recessive type of this disorder appears to be very rare; only a few cases have been reported ...
2. `medquad_ans_0000316-4` (score=0.6840) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is considered X-linked if the...
3. `medquad_ans_0000316-5` (score=0.6589) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - Genetic Testing Regis...
4. `medquad_ans_0000316-3` (score=0.6514) - Mutations in the EMD and LMNA genes cause Emery-Dreifuss muscular dystrophy. The EMD and LMNA genes provide instructions for making proteins that are components of the nuclear envelope, which surrounds the nucleus in cells. The nuclear e...
5. `medquad_ans_0000306-2` (score=0.6412) - Duchenne and Becker muscular dystrophies together affect 1 in 3,500 to 5,000 newborn males worldwide. Between 400 and 600 boys in the United States are born with these conditions each year.

**Answer**:

The provided context does not describe the symptoms of Emery-Dreifuss muscular dystrophy, X-linked. Therefore, based on the given information, the symptoms cannot be determined. [medquad_ans_0000316-2, medquad_ans_0000316-4]

**Citations detected**: `medquad_ans_0000316-2, medquad_ans_0000316-4`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000316-2, medquad_ans_0000316-4`

## Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Top docs**:

1. `medquad_ans_0000072-2` (score=0.6855) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
2. `medquad_ans_0000072-5` (score=0.6635) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins McKusick-Nathans Insti...
3. `medquad_ans_0000072-3` (score=0.6196) - Arterial tortuosity syndrome is caused by mutations in the SLC2A10 gene. This gene provides instructions for making a protein called GLUT10. The level of GLUT10 appears to be involved in the regulation of a process called the transformin...
4. `medquad_ans_0000275-2` (score=0.6149) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United States . Treatments are...
5. `medquad_ans_0000072-1` (score=0.6023) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastrointestinal tract. As i...

**Answer**:

The provided context does not mention specific treatments for arterial tortuosity syndrome. While resources for diagnosis and management are listed, including genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care, no details are given about the actual treatments used for arterial tortuosity syndrome [medquad_ans_0000072-5]. Therefore, based on the available information, the treatments for arterial tortuosity syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000072-5`
**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

## Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.6956) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence of about 1 in 2,500 i...
2. `medquad_ans_0000173-5` (score=0.6801) - These resources address the diagnosis or management of Charcot-Marie-Tooth disease: - Gene Review: Gene Review: Charcot-Marie-Tooth Hereditary Neuropathy Overview - Gene Review: Gene Review: Charcot-Marie-Tooth Neuropathy Type 1 - Gene R...
3. `medquad_ans_0000173-1` (score=0.6704) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as touch, pain, heat, and ...
4. `medquad_ans_0001104-1` (score=0.6680) - Charcot-Marie-Tooth disease type 2F (CMT2F) is a genetic disorder of the peripheral nerves. The subtypes of CMT type 2 (including type 2F) have similar features and are distinguished only by their disease-causing genes. Signs and symptom...
5. `medquad_ans_0001090-2__dup1` (score=0.6593) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1A? CMT1 is generally slowly progressive over many years. However, affected individuals often experience long periods without any obvious deterioration or progression. O...

**Answer**:

The provided context does not include information about the specific symptoms of Charcot-Marie-Tooth disease type 1C. [medquad_ans_0000173-5] lists resources related to Charcot-Marie-Tooth disease types, including type 1C, but does not describe its symptoms. Therefore, based on the given context, the symptoms of Charcot-Marie-Tooth disease type 1C cannot be determined.

**Citations detected**: `medquad_ans_0000173-5`
**Supported citations**: `medquad_ans_0000173-5`
**Unsupported citations**: _none_

## Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Top docs**:

1. `medquad_ans_0000870-1` (score=0.6490) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS damages the arteries ...
2. `medquad_ans_0000870-5` (score=0.6331) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic stenosis - MedlinePlus En...
3. `medquad_ans_0005321-1` (score=0.6191) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm with supravalvular ...
4. `medquad_ans_0000962-1` (score=0.5599) - Supravalvular aortic stenosis (SVAS) is a heart defect that develops before birth. This defect is a narrowing (stenosis) of the large blood vessel that carries blood from the heart to the rest of the body (the aorta). The condition is de...
5. `medquad_ans_0000275-2` (score=0.5544) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United States . Treatments are...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include shortness of breath, a rapid heartbeat, fatigue, and swelling in the face, feet, or abdomen. These symptoms arise due to the heart condition, supravalvular pulmonic stenosis, which impairs blood flow into the lungs, leading to reduced oxygen delivery to the body's tissues. Additionally, the retinal condition can cause exudative retinal detachment and bleeding into the eye, which may result in vision loss [medquad_ans_0000870-1].

**Citations detected**: `medquad_ans_0000870-1`
**Supported citations**: `medquad_ans_0000870-1`
**Unsupported citations**: _none_

## Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.7307) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is caused by mutations...
2. `medquad_ans_0001081-2` (score=0.6942) - 2-hydroxyglutaric aciduria is a rare disorder. D-2-HGA and L-2-HGA have each been reported to affect fewer than 150 individuals worldwide. Combined D,L-2-HGA appears to be even rarer, with only about a dozen reported cases.
3. `medquad_ans_0000067-2` (score=0.6908) - Argininosuccinic aciduria occurs in approximately 1 in 70,000 newborns.
4. `medquad_ans_0000959-2` (score=0.6791) - Approximately 350 people with succinic semialdehyde dehydrogenase deficiency have been reported worldwide.
5. `medquad_ans_0001081-5` (score=0.6769) - These resources address the diagnosis or management of 2-hydroxyglutaric aciduria: - Genetic Testing Registry: Combined d-2- and l-2-hydroxyglutaric aciduria - Genetic Testing Registry: D-2-hydroxyglutaric aciduria 1 - Genetic Testing Re...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion [medquad_ans_0001838-1].

**Citations detected**: `medquad_ans_0001838-1`
**Supported citations**: `medquad_ans_0001838-1`
**Unsupported citations**: _none_

## Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Top docs**:

1. `medquad_ans_0000943-2` (score=0.6727) - Spondylocarpotarsal synostosis syndrome is a rare disorder; its prevalence is unknown. At least 25 affected individuals have been identified.
2. `medquad_ans_0000949-5` (score=0.6644) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from MedlinePlus offer informa...
3. `medquad_ans_0000944-2` (score=0.6610) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0000949-2` (score=0.6610) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.
5. `medquad_ans_0000944-4` (score=0.6584) - Spondylocostal dysostosis can have different inheritance patterns. Types 1, 2, 3, and 4 are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with a...

**Answer**:

Yes, spondylothoracic dysostosis is inherited. It can be inherited in an autosomal recessive pattern, where both copies of the gene in each cell have mutations, or in an autosomal dominant pattern, where one copy of an altered gene in each cell is sufficient to cause the disorder. [medquad_ans_0000944-4]

**Citations detected**: `medquad_ans_0000944-4`
**Supported citations**: `medquad_ans_0000944-4`
**Unsupported citations**: _none_

## Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Top docs**:

1. `medquad_ans_0000542-2` (score=0.7227) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
2. `medquad_ans_0000542-5` (score=0.6825) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome - MedlinePlus Encycl...
3. `medquad_ans_0003296-1` (score=0.6718) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. It is inherited in an...
4. `medquad_ans_0000241-2` (score=0.6560) - Although the exact incidence is unknown, Cornelia de Lange syndrome likely affects 1 in 10,000 to 30,000 newborns. The condition is probably underdiagnosed because affected individuals with mild or uncommon features may never be recogniz...
5. `medquad_ans_0001595-5` (score=0.6521) - How is Cornelia de Lange syndrome diagnosed? A diagnosis of Cornelia de Lange syndrome (CdLS) is generally based on the presence of characteristic signs and symptoms during a thorough medical evaluation. In some cases, genetic testing ca...

**Answer**:

Jervell Lange-Nielsen syndrome is a rare form of long QT syndrome that affects an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark (at least 1 in 200,000 people). It is characterized by symptoms such as deafness from birth, arrhythmia, fainting, and sudden death. The syndrome is inherited in an autosomal recessive fashion and includes two types: type 1 and type 2 [medquad_ans_0000542-2, medquad_ans_0003296-1]. Diagnostic and management resources are available through Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia, which provide information on diagnostic tests, drug therapy, genetic counseling, and other aspects of care [medquad_ans_0000542-5].

**Citations detected**: `medquad_ans_0000542-2, medquad_ans_0003296-1`, `medquad_ans_0000542-5`
**Supported citations**: `medquad_ans_0000542-5`
**Unsupported citations**: `medquad_ans_0000542-2, medquad_ans_0003296-1`

## Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Top docs**:

1. `medquad_ans_0000218-1` (score=0.7093) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and other eye abnormali...
2. `medquad_ans_0000218-5` (score=0.7048) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Med...
3. `medquad_ans_0000042-7` (score=0.6819) - Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-related macular degenerat...
4. `medquad_ans_0000083-3__dup1` (score=0.6783) - There is no cure for Moebius syndrome. In spite of the impairments that characterize the disorder, proper care and treatment give many individuals a normal life expectancy.
5. `medquad_ans_0000184-2` (score=0.6735) - Christianson syndrome is a rare condition, although the exact prevalence is unknown. The condition was first described in a South African family and has since been found people in other parts of the world.

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy. [medquad_ans_0000218-1] describes the disorder's features and complications but does not include epidemiological data on prevalence or affected population size. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000218-1`
**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

## Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Top docs**:

1. `medquad_ans_0000040-2` (score=0.7039) - Alpha thalassemia X-linked intellectual disability syndrome appears to be a rare condition, although its exact prevalence is unknown. More than 200 affected individuals have been reported.
2. `medquad_ans_0000040-3` (score=0.7027) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although the exact function of ...
3. `medquad_ans_0000040-4` (score=0.6972) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the gene in each cell is ...
4. `medquad_ans_0000259-3__dup1` (score=0.6855) - How is alpha-thalassemia x-linked intellectual disability syndrome inherited? Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked way. The chance tha...
5. `medquad_ans_0000259-1__dup1` (score=0.6855) - Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is a genetic condition that causes intellectual disability, muscle weakness (hypotonia), short height, a particular facial appearance, genital abnormalities, and possibly...

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations alter the structure of the ATRX protein, which is essential for regulating the activity (expression) of other genes. Specifically, reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, leads to alpha thalassemia. Additionally, abnormal expression of other unidentified genes likely contributes to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Citations detected**: `medquad_ans_0000040-3`
**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

## Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Top docs**:

1. `medquad_ans_0000424-2` (score=0.7008) - GSD IV is estimated to occur in 1 in 600,000 to 800,000 individuals worldwide. Type IV accounts for roughly 3 percent of all cases of glycogen storage disease.
2. `medquad_ans_0002620-5` (score=0.6874) - How might glycogen storage disease type 4 be treated? Management of glycogen storage disease type 4 typically focuses on the signs and symptoms that are present in each individual. Studies have show that in some cases, strict dietary the...
3. `medquad_ans_0000427-5` (score=0.6774) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These resources from Medl...
4. `medquad_ans_0000421-1` (score=0.6713) - Glycogen storage disease type 0 (also known as GSD 0) is a condition caused by the body's inability to form a complex sugar called glycogen, which is a major source of stored energy in the body. GSD 0 has two types: in muscle GSD 0, glyc...
5. `medquad_ans_0002620-4` (score=0.6711) - How is glycogen storage disease type 4 inherited? Glycogen storage disease type 4 is inherited in an autosomal recessive manner. This means that an individual must have 2 abnormal copies of the GBE1 gene to be affected (one abnormal copy...

**Answer**:

The provided context does not contain specific information about the symptoms of Glycogen storage disease type 6. While it mentions resources for diagnosis and management of glycogen storage disease type VI, such as gene review and genetic testing, it does not detail the symptoms associated with this condition [medquad_ans_0000427-5]. Therefore, based on the given context, the symptoms of glycogen storage disease type 6 cannot be determined.

**Citations detected**: `medquad_ans_0000427-5`
**Supported citations**: `medquad_ans_0000427-5`
**Unsupported citations**: _none_

## Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Top docs**:

1. `medquad_ans_0000726-2` (score=0.7115) - Neurohypophyseal diabetes insipidus is thought to be rare, although its exact incidence is unknown. The acquired form occurs much more frequently than the familial form.
2. `medquad_ans_0000015-17` (score=0.7058) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone who does not have diab...
3. `medquad_ans_0000719-2` (score=0.7018) - The prevalence of nephrogenic diabetes insipidus is unknown, although the condition is thought to be rare. The acquired form occurs more frequently than the hereditary form.
4. `medquad_ans_0000033-1__dup4` (score=0.6983) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage kidney disease or ES...
5. `medquad_ans_0000016-8` (score=0.6976) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...

**Answer**:

The provided context does not specify the causes of diabetes. While it discusses complications of diabetes such as kidney disease, retinopathy, and cardiovascular issues, it does not address the underlying causes of the disease. Therefore, based on the given information, the causes of diabetes cannot be determined. [medquad_ans_0000015-17, medquad_ans_0000033-1__dup4]

**Citations detected**: `medquad_ans_0000015-17, medquad_ans_0000033-1__dup4`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000015-17, medquad_ans_0000033-1__dup4`

## Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.7088) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the laboratory and with pat...
2. `medquad_ans_0000016-24` (score=0.6848) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even blindness. Macular e...
3. `medquad_ans_0000016-6` (score=0.6829) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
4. `medquad_ans_0000016-19` (score=0.6757) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although both treatments have h...
5. `medquad_ans_0000016-8` (score=0.6724) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...

**Answer**:

Research being done for diabetic retinopathy includes studies in the laboratory and with patients to find better ways to detect, treat, and prevent vision loss in people with diabetes. Specifically, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21].

**Citations detected**: `medquad_ans_0000016-21`
**Supported citations**: `medquad_ans_0000016-21`
**Unsupported citations**: _none_

## Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.6918) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart and lung diseases wors...
2. `medquad_ans_0000007-3__dup1` (score=0.6861) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in the lungs. Severe ble...
3. `medquad_ans_0000774-1__dup1` (score=0.6796) - Respiratory failure happens when not enough oxygen passes from your lungs into your blood. Your body's organs, such as your heart and brain, need oxygen-rich blood to work well. Respiratory failure also can happen if your lungs can't rem...
4. `medquad_ans_0000651-1__dup1` (score=0.6680) - Summary : Nuclear scans use radioactive substances to see structures and functions inside your body. They use a special camera that detects radioactivity. Before the test, you receive a small amount of radioactive material. You may get i...
5. `medquad_ans_0000137-1__dup1` (score=0.6589) - Acute respiratory distress syndrome (ARDS) is a life-threatening lung condition that prevents enough oxygen from getting to the lungs and into the blood. People who develop ARDS often are very ill with another disease or have major injur...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. The best prevention is to limit exposure, especially when smoke is smelled or seen or when fires are nearby [medquad_ans_0000516-1__dup1]. Breathing in harmful fumes or smoke is also listed as a cause of ARDS, indicating its potential to cause serious respiratory injury [medquad_ans_0000007-3__dup1].

**Citations detected**: `medquad_ans_0000516-1__dup1`, `medquad_ans_0000007-3__dup1`
**Supported citations**: `medquad_ans_0000516-1__dup1`, `medquad_ans_0000007-3__dup1`
**Unsupported citations**: _none_

## Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.7542) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has a parent with MEN1 i...
2. `medquad_ans_0000134-10` (score=0.7253) - - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. - Experts do not know the exact cause of Zollinger-Ellison syndrome. - About 25 to 30 percent of gastrinomas are cause...
3. `medquad_ans_0000134-1__dup3` (score=0.7204) - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. The tumors, called gastrinomas, release large amounts of gastrin that cause the stomach to produce large amounts of acid...
4. `medquad_ans_0000134-9` (score=0.7201) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
5. `medquad_ans_0000134-3__dup2` (score=0.7000) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes hormone-releasing tumors...

**Answer**:

Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Citations detected**: `medquad_ans_0000134-4__dup2`
**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

## Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.7040) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Genetic Testing Regist...
2. `medquad_ans_0000974-2` (score=0.6893) - This condition is rare, affecting an estimated 1 in 500,000 to 1 in 1 million newborns. In most parts of the world, tetrahydrobiopterin deficiency accounts for 1 to 3 percent of all cases of elevated phenylalanine levels. The remaining c...
3. `medquad_ans_0000155-3__dup1` (score=0.6612) - How might adenylosuccinase deficiency be treated? At the current time, there are no effective therapies for the treatment of adenylosuccinase deficiency. Treatment is supportive based on the specific features.
4. `medquad_ans_0000482-1` (score=0.6607) - Histidinemia is an inherited condition characterized by elevated blood levels of the amino acid histidine, a building block of most proteins. Histidinemia is caused by the shortage (deficiency) of the enzyme that breaks down histidine. H...
5. `medquad_ans_0000898-5` (score=0.6576) - These resources address the diagnosis or management of sepiapterin reductase deficiency: - Gene Review: Gene Review: Sepiapterin Reductase Deficiency - Genetic Testing Registry: Sepiapterin reductase deficiency These resources from Medli...

**Answer**:

The provided context does not specify treatments for tetrahydrobiopterin deficiency. While it mentions resources for diagnosis and management, such as genetic testing and serum phenylalanine screening, and notes that tetrahydrobiopterin deficiency accounts for a portion of elevated phenylalanine levels, no information is given about specific therapeutic interventions [medquad_ans_0000974-5][medquad_ans_0000974-2]. Therefore, based on the available context, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Citations detected**: `medquad_ans_0000974-5`, `medquad_ans_0000974-2`
**Supported citations**: `medquad_ans_0000974-5`, `medquad_ans_0000974-2`
**Unsupported citations**: _none_

## Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.7389) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000736-4` (score=0.7062) - Nonsyndromic holoprosencephaly is inherited in an autosomal dominant pattern, which means an alteration in one copy of a gene in each cell is usually sufficient to cause the disorder. However, not all people with a gene mutation will dev...
3. `medquad_ans_0000081-3__dup1` (score=0.6969) - The prognosis for individuals with colpocephaly depends on the severity of the associated conditions and the degree of abnormal brain development. Some children benefit from special education.
4. `medquad_ans_0000189-3__dup1` (score=0.6857) - The prognosis for infants and children with megalencephaly depends upon the underlying cause and the associated neurological disorders. The prognosis for children with hemimegalencephaly is poor.
5. `medquad_ans_0000189-2__dup1` (score=0.6848) - There is no standard treatment for megalencephaly. Treatment will depend upon the disorder with which the megalencephaly is associated and will address individual symptoms and disabilities.

**Answer**:

Nonsyndromic holoprosencephaly affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2].

**Citations detected**: `medquad_ans_0000736-2`
**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

## Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Top docs**:

1. `medquad_ans_0000829-5` (score=0.6515) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholestasis 2 - Genetic Tes...
2. `medquad_ans_0000181-13` (score=0.6509) - - Alagille syndrome is a genetic condition that results in various symptoms in different parts of the body, including the liver. - A person with Alagille syndrome has fewer than the normal number of small bile ducts inside the liver. - I...
3. `medquad_ans_0000525-3` (score=0.6329) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt export pump (BSEP). Th...
4. `medquad_ans_0000118-5` (score=0.6320) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 - Genetic Testing Reg...
5. `medquad_ans_0003240-3` (score=0.6277) - What causes intrahepatic cholestasis of pregnancy? Largely, the cause of intrahepatic cholestasis of pregnancy (ICP) is unknown. ICP is present in approximately 1% of pregnancies in the United States. It is thought to be caused by a mixt...

**Answer**:

The provided context does not specify the exact genetic changes related to progressive familial intrahepatic cholestasis. It mentions resources that address the diagnosis or management of progressive familial intrahepatic cholestasis, including genetic testing registry entries for types 2 and 3, and references to ATP8B1 deficiency, which is associated with a specific form of the condition [medquad_ans_0000829-5]. However, no detailed description of the specific genetic mutations or changes is included in the context. Therefore, based on the given information, the genetic changes related to progressive familial intrahepatic cholestasis cannot be determined.

**Citations detected**: `medquad_ans_0000829-5`
**Supported citations**: `medquad_ans_0000829-5`
**Unsupported citations**: _none_

## Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Top docs**:

1. `medquad_ans_0000631-2` (score=0.6647) - McCune-Albright syndrome occurs in between 1 in 100,000 and 1 in 1,000,000 people worldwide.
2. `medquad_ans_0000152-2` (score=0.6494) - Carney complex is a rare disorder; fewer than 750 affected individuals have been identified.
3. `medquad_ans_0000290-2` (score=0.6492) - Distal arthrogryposis type 1 affects an estimated 1 in 10,000 people worldwide.
4. `medquad_ans_0000564-2` (score=0.6488) - Kniest dysplasia is a rare condition; the exact incidence is unknown.
5. `medquad_ans_0000438-2` (score=0.6482) - Greenberg dysplasia is a very rare condition. Approximately ten cases have been reported in the scientific literature.

**Answer**:

The provided context does not contain any information about the symptoms of Smith McCort dysplasia. Therefore, the question cannot be answered based on the given context. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.6970) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive disease is a condit...
2. `medquad_ans_0000453-5` (score=0.6917) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodeficiency These resources...
3. `medquad_ans_0000851-2` (score=0.6228) - Purine nucleoside phosphorylase deficiency is rare; only about 70 affected individuals have been identified. This disorder accounts for approximately 4 percent of all SCID cases.
4. `medquad_ans_0000530-2` (score=0.6211) - IBD deficiency is a rare disorder; approximately 22 cases have been reported in the medical literature.
5. `medquad_ans_0000407-2` (score=0.6186) - Gilbert syndrome is a common condition that is estimated to affect 3 to 7 percent of Americans.

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency (VODI) is inherited. This is supported by the context stating that it is "a hereditary disorder of the liver and immune system" [medquad_ans_0000453-1].

**Citations detected**: `medquad_ans_0000453-1`
**Supported citations**: `medquad_ans_0000453-1`
**Unsupported citations**: _none_

## Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Top docs**:

1. `medquad_ans_0000347-2` (score=0.6669) - Familial erythrocytosis is a rare condition; its prevalence is unknown.
2. `medquad_ans_0005902-1` (score=0.6599) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how often the symptom is ...
3. `medquad_ans_0000476-2` (score=0.6568) - Hereditary spherocytosis occurs in 1 in 2,000 individuals of Northern European ancestry. This condition is the most common cause of inherited anemia in that population. The prevalence of hereditary spherocytosis in people of other ethnic...
4. `medquad_ans_0001062-2` (score=0.6565) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
5. `medquad_ans_0000307-2` (score=0.6528) - Dyserythropoietic anemia and thrombocytopenia is a rare condition; its prevalence is unknown. Occasionally, individuals with this disorder are mistakenly diagnosed as having more common blood disorders, making it even more difficult to d...

**Answer**:

The signs and symptoms of Stomatocytosis I include autosomal dominant inheritance, hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis [medquad_ans_0005902-1].

**Citations detected**: `medquad_ans_0005902-1`
**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

## Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Top docs**:

1. `medquad_ans_0000159-1__dup3` (score=0.6762) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 percent of people with...
2. `medquad_ans_0000159-8` (score=0.6475) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed through urine and b...
3. `medquad_ans_0000186-3__dup1` (score=0.6332) - The prognosis for lupus varies widely depending on the organs involved and the intensity of the inflammatory reaction. The course of lupus is commonly chronic and relapsing, often with long periods of remission. Most individuals with lup...
4. `medquad_ans_0000159-7` (score=0.6330) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may need to make to their...
5. `medquad_ans_0003688-1` (score=0.6297) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, high blood pressure, ...

**Answer**:

Treatment for lupus nephritis includes medications that suppress the immune system to prevent damage to the kidneys. Standard treatment typically involves a corticosteroid, such as prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication like cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. People with lupus nephritis causing high blood pressure may also need medications to lower blood pressure, which can slow the progression of kidney disease. In many cases, treatment is effective in controlling the condition, though 10 to 30 percent of people may still develop kidney failure despite treatment. Additionally, individuals with kidney disease progressing due to lupus nephritis may need to consult a healthcare provider about dietary changes, such as reducing sodium intake, especially if they have high blood pressure [medquad_ans_0000159-8].

**Citations detected**: `medquad_ans_0000159-8`
**Supported citations**: `medquad_ans_0000159-8`
**Unsupported citations**: _none_

## Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Top docs**:

1. `medquad_ans_0000246-3__dup1` (score=0.6773) - RLS is generally a life-long condition for which there is no cure. Symptoms may gradually worsen with age. Nevertheless, current therapies can control the disorder, minimizing symptoms and increasing periods of restful sleep. In addition...
2. `medquad_ans_0000869-5` (score=0.6741) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndrome, susceptibility t...
3. `medquad_ans_0000246-1__dup2` (score=0.6703) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the disorder often desc...
4. `medquad_ans_0000869-2` (score=0.6700) - Restless legs syndrome is one of the most common sleep and movement disorders. It affects an estimated 5 to 10 percent of adults and 2 to 4 percent of children in the United States. For unknown reasons, the disorder affects women more of...
5. `medquad_ans_0005317-1` (score=0.6693) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sitting or lying down a...

**Answer**:

Current therapies can control restless legs syndrome, minimizing symptoms and increasing periods of restful sleep [medquad_ans_0000246-3__dup1]. Treatment options include drug therapy, which is one of the resources provided by MedlinePlus for managing the condition [medquad_ans_0000869-5]. The National Heart, Lung, and Blood Institute offers specific information on how restless legs syndrome is treated [medquad_ans_0000869-5]. Additionally, the Restless Leg Syndrome Foundation provides information on treatment options [medquad_ans_0000869-5].

**Citations detected**: `medquad_ans_0000246-3__dup1`, `medquad_ans_0000869-5`
**Supported citations**: `medquad_ans_0000246-3__dup1`, `medquad_ans_0000869-5`
**Unsupported citations**: _none_

## Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Top docs**:

1. `medquad_ans_0002573-6` (score=0.6594) - How might glioblastoma be treated? Unfortunately, there is no cure for glioblastoma. Treatment is palliative and may include surgery, radiation therapy and/or chemotherapy. The best treatment options for each person depend on many factor...
2. `medquad_ans_0002573-3` (score=0.6328) - What causes glioblastoma? In most cases, the exact underlying cause of glioblastoma is unknown. In rare cases, they can occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome and Li Fraumeni synd...
3. `medquad_ans_0002573-4` (score=0.6278) - Is glioblastoma inherited? Most glioblastomas are not inherited. They usually occur sporadically in people with no family history of tumors. However, they can rarely occur in people with certain genetic syndromes such as neurofibromatosi...
4. `medquad_ans_0000006_1-3` (score=0.6199) - The cause of most adult brain and spinal cord tumors is not known.
5. `medquad_ans_0002573-2` (score=0.6125) - What are the signs and symptoms of glioblastoma? Signs and symptoms of glioblastoma vary depending on the size and location of the tumor but may include: Headache Nausea and vomiting Drowsiness Changes in personality Weakness on one side...

**Answer**:

Glioblastoma is a type of brain tumor that is typically aggressive and has no known cure. It is often treated with surgery, radiation therapy, and/or chemotherapy, depending on factors such as tumor size, location, extent of growth into surrounding brain tissue, and the patient's age and overall health. In most cases, it is not possible to remove the entire tumor, so additional treatments are necessary. The exact underlying cause of glioblastoma is unknown in most cases, though it can occasionally occur in individuals with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome, and Li Fraumeni syndrome, which are inherited in an autosomal dominant manner. Most glioblastomas are not inherited and occur sporadically without a family history of tumors [medquad_ans_0002573-6, medquad_ans_0002573-3, medquad

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Top docs**:

1. `medquad_ans_0000251-3__dup1` (score=0.6917) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
2. `medquad_ans_0000251-4__dup1` (score=0.6855) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to prevent or treat develo...
3. `medquad_ans_0005496-4` (score=0.6699) - Is schizencephaly inherited? Schizencephaly is not thought to be inherited in most cases and it rarely affects more than one person in a family. A few cases of familial schizencephaly have been linked to changes (mutations) in the EMX2 g...
4. `medquad_ans_0000736-2` (score=0.6647) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
5. `medquad_ans_0000189-3__dup1` (score=0.6577) - The prognosis for infants and children with megalencephaly depends upon the underlying cause and the associated neurological disorders. The prognosis for children with hemimegalencephaly is poor.

**Answer**:

The provided context does not specify the exact causes of schizencephaly. It mentions that schizencephaly is not thought to be inherited in most cases and that a few cases of familial schizencephaly have been linked to changes (mutations) in the EMX2 gene [medquad_ans_0005496-4]. However, no comprehensive or detailed explanation of the causes is given in the context. Therefore, based on the available information, the specific causes of schizencephaly cannot be determined.

**Citations detected**: `medquad_ans_0005496-4`
**Supported citations**: `medquad_ans_0005496-4`
**Unsupported citations**: _none_

## Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0000309-4__dup1` (score=0.6842) - Is amyotrophic lateral sclerosis (ALS) inherited? About 90-95% percent of cases of ALS are not inherited and occur in individuals with no history of the condition in their family. The remaining 5-10% of cases are familial, and are though...
2. `medquad_ans_0000052-4` (score=0.6770) - About 90 to 95 percent of ALS cases are sporadic, which means they are not inherited. An estimated 5 to 10 percent of ALS is familial and caused by mutations in one of several genes. The pattern of inheritance varies depending on the gen...
3. `medquad_ans_0000551-4` (score=0.6744) - When caused by mutations in the ALS2 gene, juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal...
4. `medquad_ans_0000052-1` (score=0.6721) - Amyotrophic lateral sclerosis (ALS) is a progressive disease that affects motor neurons, which are specialized nerve cells that control muscle movement. These nerve cells are found in the spinal cord and the brain. In ALS, motor neurons ...
5. `medquad_ans_0000551-3` (score=0.6668) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully understood. Mutations...

**Answer**:

The genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in one or more genes that are associated with familial ALS. Approximately 5-10% of ALS cases are familial and are caused by mutations in several genes. Most familial cases follow an autosomal dominant inheritance pattern, meaning that only one mutated copy of the disease-causing gene in each cell is sufficient to cause the condition [medquad_ans_0000309-4__dup1]. In rare cases, ALS may be inherited in an autosomal recessive manner, requiring mutations in both copies of the disease-causing gene (one inherited from each parent), or in an X-linked dominant manner, where the disease-causing gene is located on the X chromosome [medquad_ans_0000309-4__dup1]. Some individuals who inherit these mutations may never develop signs or symptoms, a phenomenon known as reduced penetrance [med

**Citations detected**: `medquad_ans_0000309-4__dup1`
**Supported citations**: `medquad_ans_0000309-4__dup1`
**Unsupported citations**: _none_

## Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Top docs**:

1. `medquad_ans_0003473-6` (score=0.7406) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injections of insulin-like g...
2. `medquad_ans_0003473-5` (score=0.7339) - How is Laron syndrome diagnosed? A diagnosis of Laron syndrome is often suspected based on the presence of characteristic signs and symptoms. Additional testing can then be ordered to confirm the diagnosis and rule out other conditions t...
3. `medquad_ans_0000581-2` (score=0.7242) - Laron syndrome is a rare disorder. About 350 people have been diagnosed with the condition worldwide. The largest single group of affected individuals (about 100 people) lives in an area of southern Ecuador.
4. `medquad_ans_0003473-1` (score=0.7201) - Laron syndrome is a condition that occurs when the body is unable to utilize growth hormone. It is primarily characterized by short stature. Other signs and symptoms vary but may include reduced muscle strength and endurance; hypoglycemi...
5. `medquad_ans_0000581-4` (score=0.7156) - Most cases of Laron syndrome are inherited in an autosomal recessive pattern, which means both copies of the GHR gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of ...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and growth hormone (GH) levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6]. Treatment is primarily focused on improving growth [medquad_ans_0003473-1].

**Citations detected**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Supported citations**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Unsupported citations**: _none_

## Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-24` (score=0.6505) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even blindness. Macular e...
2. `medquad_ans_0000016-6` (score=0.6486) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
3. `medquad_ans_0000016-19` (score=0.6466) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although both treatments have h...
4. `medquad_ans_0000016-21` (score=0.6358) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the laboratory and with pat...
5. `medquad_ans_0000016-8` (score=0.6328) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...

**Answer**:

Treatments for diabetic retinopathy include timely intervention and appropriate follow-up care, which can reduce the risk of blindness by 95 percent in people with proliferative retinopathy [medquad_ans_0000016-19]. Specific treatments may involve laser surgery, and researchers are also studying drugs that could reduce the need for laser surgery [medquad_ans_0000016-21]. However, these treatments do not cure diabetic retinopathy, and individuals remain at risk for new bleeding and may require repeated treatment to protect their vision [medquad_ans_0000016-19].

**Citations detected**: `medquad_ans_0000016-19`, `medquad_ans_0000016-21`
**Supported citations**: `medquad_ans_0000016-19`, `medquad_ans_0000016-21`
**Unsupported citations**: _none_

## Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Top docs**:

1. `medquad_ans_0001421-1` (score=0.6691) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; the inflammation can...
2. `medquad_ans_0001421-2` (score=0.6510) - What are the signs and symptoms of collagenous colitis? All individuals with collagenous colitis experience chronic, watery, non-bloody diarrhea which is what typically prompts individuals to seek medical attention. Onset of diarrhea may...
3. `medquad_ans_0000116-11` (score=0.6318) - - Microscopic colitis is an inflammation of the colon that a health care provider can see only with a microscope. - The two types of microscopic colitis are collagenous colitis and lymphocytic colitis. - The exact cause of microscopic co...
4. `medquad_ans_0000122-1__dup3` (score=0.6282) - Ulcerative colitis is a chronic, or long lasting, disease that causes inflammationirritation or swellingand sores called ulcers on the inner lining of the large intestine. Ulcerative colitis is a chronic inflammatory disease of the gastr...
5. `medquad_ans_0000116-1__dup4` (score=0.6235) - Microscopic colitis is an inflammation of the colon that a health care provider can see only with a microscope. Inflammation is the bodys normal response to injury, irritation, or infection of tissues. Microscopic colitis is a type of in...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each individual. In some cases, the condition resolves on its own. The gastroenterologist may prescribe medications to help control symptoms. Medications are almost always effective in treating microscopic colitis, which includes collagenous colitis. The gastroenterologist may also recommend eating, diet, and nutrition changes [medquad_ans_0001421-1][medquad_ans_0000116-11].

**Citations detected**: `medquad_ans_0001421-1`, `medquad_ans_0000116-11`
**Supported citations**: `medquad_ans_0001421-1`, `medquad_ans_0000116-11`
**Unsupported citations**: _none_

## Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Top docs**:

1. `medquad_ans_0000418-6` (score=0.6469) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision to treat based upon...
2. `medquad_ans_0000418-2__dup1` (score=0.6419) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
3. `medquad_ans_0000418-1__dup2` (score=0.6412) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (meat-eating) animals...
4. `medquad_ans_0000432-1__dup3` (score=0.6405) - Whipworm (Trichuris trichiura) is an intestinal parasite of humans. The larvae and adult worms live in the intestine of humans and can cause intestinal disease. The name is derived from the worm’s distinctive whip-like shape.
5. `medquad_ans_0000418-5__dup1` (score=0.6357) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild game meat. Laborato...

**Answer**:

To prevent trichinellosis (also known as trichinosis), avoid consuming raw or undercooked meat, particularly from wild game such as bear or cougar, or from domestic pigs and wild boar. Proper cooking of meat to safe internal temperatures kills Trichinella parasites. Additionally, commercial and home freezing of pork can reduce the risk of infection. Public awareness of the dangers of eating raw or undercooked meat products also plays a key role in prevention [medquad_ans_0000418-2__dup1].

**Citations detected**: `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Top docs**:

1. `medquad_ans_0000536-1` (score=0.7135) - Autoimmune hepatitis is a disease in which the bodys immune system attacks liver cells. This immune response causes inflammation of the liver, also called hepatitis. The disease can be quite serious and, if not treated, gets worse over t...
2. `medquad_ans_0000536-3` (score=0.7088) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic attack. This autoimmun...
3. `medquad_ans_0000536-4` (score=0.7007) - How is autoimmune hepatitis diagnosed? The diagnosis of autoimmune hepatitis is typically made based on symptoms, blood tests, and a liver biopsy.
4. `medquad_ans_0000182-3__dup1` (score=0.6901) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
5. `medquad_ans_0000182-1__dup3` (score=0.6891) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally protects people from inf...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition. The exact cause is unknown, but evidence suggests that liver injury results from a cell-mediated immunologic attack, which may be triggered by genetic factors, viral infections, or chemical agents. Autoimmune hepatitis sometimes occurs in relatives of people with autoimmune diseases, further suggesting a genetic cause [medquad_ans_0000536-3][medquad_ans_0000182-3__dup1].

**Citations detected**: `medquad_ans_0000536-3`, `medquad_ans_0000182-3__dup1`
**Supported citations**: `medquad_ans_0000536-3`, `medquad_ans_0000182-3__dup1`
**Unsupported citations**: _none_

## Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Top docs**:

1. `medquad_ans_0000006_3-2` (score=0.6680) - The cause of most childhood brain tumors is not known.
2. `medquad_ans_0000006_9-2` (score=0.6673) - The cause of most childhood brain tumors is unknown.
3. `medquad_ans_0000006_5-2` (score=0.6673) - The cause of most childhood brain tumors is unknown.
4. `medquad_ans_0000006_3-1` (score=0.6548) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central nervous system contr...
5. `medquad_ans_0000006_3-9` (score=0.6523) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treating childhood brain ...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. Astrocytomas may be benign or malignant and arise from star-shaped brain cells called astrocytes, which are a type of glial cell. These cells support and protect nerve cells by holding them in place, bringing them food and oxygen, and shielding them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Citations detected**: `medquad_ans_0000006_3-1`
**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

## Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Top docs**:

1. `medquad_ans_0000013_3-4__dup2` (score=0.6549) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL1 negative. For more...
2. `medquad_ans_0000013_3-1__dup3` (score=0.6520) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising ...
3. `medquad_ans_0000040-10` (score=0.6304) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
4. `medquad_ans_0000013_3-2__dup3` (score=0.6259) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check with your doctor if ...
5. `medquad_ans_0000013_3-1__dup1` (score=0.6257) - Key Points - Myelodysplastic/myeloproliferative neoplasm, unclassifiable, is a disease that has features of both myelodysplastic and myeloproliferative diseases but is not chronic myelomonocytic leukemia, juvenile myelomonocytic leukemia...

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become a type of white blood cell called granulocytes, and some of these stem cells never become mature white blood cells. These immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Atypical CML is similar to CML in that the leukemia cells look alike under a microscope, but it does not have the "Philadelphia chromosome" chromosome change that is typically present in CML [medquad_ans_0000013_3-1__dup3].

**Citations detected**: `medquad_ans_0000013_3-1__dup3`
**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

## Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.6821) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the characteristic features...
2. `medquad_ans_0000996-2` (score=0.6759) - Trisomy 13 occurs in about 1 in 16,000 newborns. Although women of any age can have a child with trisomy 13, the chance of having a child with this condition increases as a woman gets older.
3. `medquad_ans_0000996-5` (score=0.6575) - These resources address the diagnosis or management of trisomy 13: - Genetic Testing Registry: Complete trisomy 13 syndrome - MedlinePlus Encyclopedia: Trisomy 13 These resources from MedlinePlus offer information about the diagnosis and...
4. `medquad_ans_0000996-4` (score=0.6557) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with an abnormal number of...
5. `medquad_ans_0000303-3` (score=0.6439) - Most cases of Down syndrome result from trisomy 21, which means each cell in the body has three copies of chromosome 21 instead of the usual two copies. Less commonly, Down syndrome occurs when part of chromosome 21 becomes attached (tra...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In rare cases, trisomy 13 can result from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals may have two normal copies of chromosome 13 and an extra copy attached to another chromosome. In very rare instances, only part of chromosome 13 is

**Citations detected**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

## Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.7066) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - Genetic Testing Regis...
2. `medquad_ans_0000218-2__dup1` (score=0.6986) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bone marrow transplant...
3. `medquad_ans_0000218-4__dup1` (score=0.6873) - The National Institute of Neurological Disorders and Stroke (NINDS), a part of the National Institutes of Health (NIH), conducts and supports research about Niemann-Pick disease through research grants to research institutions across the...
4. `medquad_ans_0004415-3` (score=0.6785) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respiratory problems A cardio...
5. `medquad_ans_0000014-2__dup1` (score=0.6715) - There is no cure for Alexander disease, nor is there a standard course of treatment. Treatment of Alexander disease is symptomatic and supportive.

**Answer**:

There is currently no cure for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists for gastrointestinal issues. Learning specialists may also be involved to support cognitive development. For Niemann-Pick disease type B, bone marrow transplantation has been attempted in a few individuals, and enzyme replacement and gene therapies might be helpful. However, restricting one's diet does not prevent the buildup of lipids in cells and tissues. There is no effective treatment for persons with type A [medquad_ans_0000218-2__dup1]. Research is ongoing to identify biomarkers and potential treatments [medquad_ans_0000218-4__dup1].

**Citations detected**: `medquad_ans_0000218-2__dup1`, `medquad_ans_0000218-4__dup1`
**Supported citations**: `medquad_ans_0000218-2__dup1`, `medquad_ans_0000218-4__dup1`
**Unsupported citations**: _none_

## Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.6533) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no oth...
2. `medquad_ans_0001714-1` (score=0.6457) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial polydactyly. If the info...
3. `medquad_ans_0000270-2` (score=0.6380) - Dandy-Walker malformation is estimated to affect 1 in 10,000 to 30,000 newborns.
4. `medquad_ans_0003799-3` (score=0.6322) - What causes Marden-Walker syndrome? The underlying cause of Marden-Walker syndrome has not been clearly established. It appears to be a developmental disorder of the central nervous system and is likely to be the expression of various he...
5. `medquad_ans_0000270-4` (score=0.6300) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malformation does not have...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [1]. The frequencies listed are

**Citations detected**: `1`
**Supported citations**: _none_
**Unsupported citations**: `1`

## Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Top docs**:

1. `medquad_ans_0003789-1` (score=0.6414) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgkin lymphomas and is ...
2. `medquad_ans_0003789-4` (score=0.6353) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be performed, but the diagn...
3. `medquad_ans_0003789-5` (score=0.6284) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance stagestreatment includ...
4. `medquad_ans_0000698-2` (score=0.5958) - Mycosis fungoides occurs in about 1 in 100,000 to 350,000 individuals. It accounts for approximately 70 percent of cutaneous T-cell lymphomas. For unknown reasons, mycosis fungoides affects males nearly twice as often as females. In the ...
5. `medquad_ans_0001025-4` (score=0.5934) - Waldenstrm macroglobulinemia is usually not inherited, and most affected people have no history of the disorder in their family. The condition usually arises from mutations that are acquired during a person's lifetime (somatic mutations)...

**Answer**:

The provided context does not specify the exact cause of mantle cell lymphoma. It mentions that mantle cell lymphoma is a B-cell lymphoma that develops from cancerous B-cells in the mantle zone of the lymph node and that chromosome translocations can be detected via genetic molecular testing, but it does not state what causes these translocations or the underlying etiology of the disease. Therefore, based on the given information, the cause of mantle cell lymphoma cannot be determined.

[medquad_ans_0003789-1]

**Citations detected**: `medquad_ans_0003789-1`
**Supported citations**: `medquad_ans_0003789-1`
**Unsupported citations**: _none_

## Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Top docs**:

1. `medquad_ans_0000589-2` (score=0.7325) - The incidence of Legg-Calv-Perthes disease varies by population. The condition is most common in white populations, in which it affects an estimated 1 to 3 in 20,000 children under age 15.
2. `medquad_ans_0000589-1` (score=0.6868) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically between ages 4 and ...
3. `medquad_ans_0000589-5` (score=0.6853) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosis and management of ...
4. `medquad_ans_0000589-3` (score=0.6796) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg-Calv-Perthes diseas...
5. `medquad_ans_0000757-2` (score=0.6616) - This condition affects an estimated 6 to 7 per 100,000 people worldwide. Types I and IV are the most common forms of osteogenesis imperfecta, affecting 4 to 5 per 100,000 people.

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. However, in a small percentage of cases, mutations in the COL2A1 gene are associated with the bone abnormalities characteristic of the disease [medquad_ans_0000589-3]. Therefore, while it is not typically inherited, there can be a genetic component in rare instances.

**Citations detected**: `medquad_ans_0000589-3`
**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

## Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Top docs**:

1. `medquad_ans_0003965-3` (score=0.7047) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for an individual to be ...
2. `medquad_ans_0000647-5` (score=0.6836) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Registry: Sphingolipid...
3. `medquad_ans_0000181-3__dup1` (score=0.6744) - The prognosis for the leukodystrophies varies according to the specific type of leukodystrophy.
4. `medquad_ans_0000647-2` (score=0.6743) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigrated to Israel from s...
5. `medquad_ans_0000192-1__dup1` (score=0.6715) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage materials in cells i...

**Answer**:

Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage materials in cells in the white matter of the central nervous system and peripheral nerves. The buildup of storage materials impairs the growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fibers. MLD is one of several lipid storage diseases, which result in the harmful buildup of lipids in brain cells and other cells and tissues in the body [medquad_ans_0000192-1__dup1]. It is inherited in an autosomal recessive manner, meaning both copies of the disease-causing gene in each cell must have a mutation for an individual to be affected [medquad_ans_0003965-

**Citations detected**: `medquad_ans_0000192-1__dup1`
**Supported citations**: `medquad_ans_0000192-1__dup1`
**Unsupported citations**: _none_

## Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Top docs**:

1. `medquad_ans_0000401-2` (score=0.6299) - Geleophysic dysplasia is a rare disorder whose prevalence is unknown. More than 30 affected individuals have been reported.
2. `medquad_ans_0000648-2` (score=0.6276) - Metatropic dysplasia is a rare disease; its exact prevalence is unknown. More than 80 affected individuals have been reported in the scientific literature.
3. `medquad_ans_0000144-2` (score=0.6245) - The prevalence of campomelic dysplasia is uncertain; estimates range from 1 in 40,000 to 200,000 people.
4. `medquad_ans_0000010-2` (score=0.6192) - Acromicric dysplasia is a rare disorder; its prevalence is unknown.
5. `medquad_ans_0000831-5` (score=0.6183) - These resources address the diagnosis or management of progressive pseudorheumatoid dysplasia: - Cedars-Sinai: Skeletal Dysplasias - Gene Review: Gene Review: Progressive Pseudorheumatoid Dysplasia - Genetic Testing Registry: Progressive...

**Answer**:

The provided context does not contain information about the symptoms of pseudodiastrophic dysplasia. [medquad_ans_0000831-5] mentions resources for progressive pseudorheumatoid dysplasia, but there is no mention of pseudodiastrophic dysplasia or its symptoms. Therefore, the requested information is not available in the given context.

**Citations detected**: `medquad_ans_0000831-5`
**Supported citations**: `medquad_ans_0000831-5`
**Unsupported citations**: _none_

## Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Top docs**:

1. `medquad_ans_0000251-2` (score=0.7237) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
2. `medquad_ans_0001664-4` (score=0.7060) - Is cri du chat syndrome inherited? Most cases of cri du chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Most affe...
3. `medquad_ans_0000251-4` (score=0.7023) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typically have no history of...
4. `medquad_ans_0001664-3` (score=0.6865) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals but stu...
5. `medquad_ans_0000251-3` (score=0.6858) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that larger deletions te...

**Answer**:

Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. The size of the deletion varies among affected individuals, with larger deletions tend to result in more severe intellectual disability and developmental delay than smaller deletions. Most cases are not inherited and occur as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected individuals typically have no family history of the disorder. About 10 percent of people with cri-du-chat syndrome inherit the chromosome abnormality from an unaffected parent who carries a balanced translocation, which usually does not cause health problems but can become unbalanced and lead to missing genetic material, resulting in the intellectual disability and health problems characteristic of the disorder [medquad_ans_0001664-4]. The condition occurs in an estimated 1 in 20,000 to

**Citations detected**: `medquad_ans_0001664-4`
**Supported citations**: `medquad_ans_0001664-4`
**Unsupported citations**: _none_

## Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000720-1__dup1` (score=0.6470) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the entire surface of the br...
2. `medquad_ans_0004956-1` (score=0.6462) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associated with the condition...
3. `medquad_ans_0000807-1` (score=0.6436) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops too many folds, and...
4. `medquad_ans_0000722-1__dup1` (score=0.6418) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of the face, tongue, jaws...
5. `medquad_ans_0000807-5` (score=0.6381) - These resources address the diagnosis or management of polymicrogyria: - Gene Review: Gene Review: Polymicrogyria Overview - Genetic Testing Registry: Congenital bilateral perisylvian syndrome - Genetic Testing Registry: Polymicrogyria, ...

**Answer**:

The provided context does not contain specific information about the symptoms of Bilateral frontoparietal polymicrogyria. While it describes general symptoms of bilateral polymicrogyria, such as severe intellectual disability, problems with movement, seizures, and delayed development, it does not specify symptoms related to the frontoparietal region. Therefore, based on the given context, the symptoms of Bilateral frontoparietal polymicrogyria cannot be determined. [medquad_ans_0000720-1, medquad_ans_0004956-1, medquad_ans_0000807-1]

**Citations detected**: `medquad_ans_0000720-1, medquad_ans_0004956-1, medquad_ans_0000807-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000720-1, medquad_ans_0004956-1, medquad_ans_0000807-1`

## Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Top docs**:

1. `medquad_ans_0003414-4` (score=0.7297) - Is genetic testing available for Konigsmark Knox Hussels syndrome? GeneTests lists the names of laboratories that are performing genetic testing for Konigsmark Knox Hussels syndrome. To view the contact information for the clinical labor...
2. `medquad_ans_0000565-2` (score=0.7271) - Knobloch syndrome is a rare condition. However, the exact prevalence of the condition is unknown.
3. `medquad_ans_0003414-1` (score=0.7042) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, which results in mildly ...
4. `medquad_ans_0000562-2` (score=0.6976) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
5. `medquad_ans_0000553-2` (score=0.6954) - Kabuki syndrome occurs in approximately 1 in 32,000 newborns.

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene and is inherited in an autosomal dominant fashion [medquad_ans_0003414-1].

**Citations detected**: `medquad_ans_0003414-1`
**Supported citations**: `medquad_ans_0003414-1`
**Unsupported citations**: _none_

## Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Top docs**:

1. `medquad_ans_0004589-4` (score=0.6987) - How is orofaciodigital syndrome type 2 inherited? Orofaciodigital syndrome type 2 is inherited in an autosomal recessive pattern, which means that an individual needs to inherit two mutated (changed) copies of the gene-one from each pare...
2. `medquad_ans_0004589-3` (score=0.6862) - What causes orofaciodigital syndrome type 2? Orofaciodigital syndrome type 2 is caused by mutations (changes) of an as yet unidentified gene.
3. `medquad_ans_0000754-2` (score=0.6858) - Oral-facial-digital syndrome has an estimated incidence of 1 in 50,000 to 250,000 newborns. Type I accounts for the majority of cases of this disorder. The other forms of oral-facial-digital syndrome are very rare; most have been identif...
4. `medquad_ans_0000761-2` (score=0.6844) - Otopalatodigital syndrome type 1 is a rare disorder, affecting fewer than 1 in every 100,000 individuals. Its specific incidence is unknown.
5. `medquad_ans_0004584-3` (score=0.6799) - Is genetic testing available for orofaciodigital syndrome 1 (OFD1)? Genetic testing for orofaciodigital syndrome 1 is clinically available. OFD1 is the only gene currently known to be associated with this condition. Testing is often used...

**Answer**:

The provided context does not contain information about the symptoms of Orofaciodigital syndrome 1. [medquad_ans_0004589-4, medquad_ans_0004589-3, medquad_ans_0000754-2, medquad_ans_0000761-2, medquad_ans_0004584-3] discuss inheritance, causes, incidence, and genetic testing but do not list symptoms of Orofaciodigital syndrome 1. Therefore, the symptoms cannot be determined from the given context.

**Citations detected**: `medquad_ans_0004589-4, medquad_ans_0004589-3, medquad_ans_0000754-2, medquad_ans_0000761-2, medquad_ans_0004584-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0004589-4, medquad_ans_0004589-3, medquad_ans_0000754-2, medquad_ans_0000761-2, medquad_ans_0004584-3`

## Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Top docs**:

1. `medquad_ans_0000694-1__dup1` (score=0.7672) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of medical errors, you ca...
2. `medquad_ans_0000018-1__dup2` (score=0.7639) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects to expect. There can ...
3. `medquad_ans_0000193-6` (score=0.7575) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need to know about Liver T...
4. `medquad_ans_0000869-1__dup1` (score=0.7463) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can bring: - Lists of your c...
5. `medquad_ans_0000863-1__dup1` (score=0.7424) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For example, a surgeon may ...

**Answer**:

Yes, there is information about surgery. After any operation, patients may experience side effects such as pain, swelling, and soreness in the area where the surgery was performed. The surgeon can provide information on expected side effects and potential complications, which may include infection, excessive bleeding, reaction to anesthesia, or accidental injury. Individuals with certain medical conditions may be at a higher risk for complications. Patients should also ask their surgeon about how long they will be in the hospital, what supplies or equipment they may need at home, when they can return to work, when it is safe to resume exercising, and any activity restrictions. Following the surgeon's advice can help facilitate a faster recovery [medquad_ans_0000018-1__dup2].

**Citations detected**: `medquad_ans_0000018-1__dup2`
**Supported citations**: `medquad_ans_0000018-1__dup2`
**Unsupported citations**: _none_

## Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Top docs**:

1. `medquad_ans_0002935-4` (score=0.6829) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspected or diagnosed since...
2. `medquad_ans_0002935-3` (score=0.6826) - What causes human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-2 is spread by blood transfusions, sexual contact and sharing ne...
3. `medquad_ans_0002935-5` (score=0.6551) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to others. Screening blo...
4. `medquad_ans_0002935-1` (score=0.6549) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may late...
5. `medquad_ans_0002934-3` (score=0.6243) - What causes human T-cell leukemia virus, type 1? Human T-cell leukemia virus, type 1 (HTLV-1) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-1 is spread by blood transfusions, sexual contact and sharing ne...

**Answer**:

Human T-cell leukemia virus type 2 (HTLV-2) is a retroviral infection that affects T cells, a type of white blood cell. Although it generally causes no signs or symptoms, some individuals may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread through blood transfusions, sexual contact, sharing needles, and from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic throughout life [medquad_ans_0002935-1]. Diagnosis is typically based on blood tests that detect antibodies to the virus, and it may occur during blood donation screening, due to a family history of infection, or as part of a work-up for HTLV-2-associated medical problems [medquad_ans_0002935-4]. It is unclear why some

**Citations detected**: `medquad_ans_0002935-1`, `medquad_ans_0002935-4`
**Supported citations**: `medquad_ans_0002935-1`, `medquad_ans_0002935-4`
**Unsupported citations**: _none_

## Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Top docs**:

1. `medquad_ans_0001152-1` (score=0.7203) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD decrease a child's ab...
2. `medquad_ans_0001152-2` (score=0.6952) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment for chILD is to rel...
3. `medquad_ans_0000406-5__dup1` (score=0.6822) - What treatment is available for antisynthetase syndrome? Corticosteroids are typically the first-line of treatment and may be required for several months or years. These medications are often given orally; however, in severe cases, intra...
4. `medquad_ans_0000853-1__dup1` (score=0.6822) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infections, in associatio...
5. `medquad_ans_0000522-1__dup1` (score=0.6774) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other pa...

**Answer**:

Interstitial Lung Diseases (ILDs) are a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis [medquad_ans_0000522-1__dup1]. These diseases affect infants, children, and adolescents and reduce the lungs' ability to exchange oxygen and carbon dioxide, sometimes causing fluid and other materials to collect in the lungs [medquad_ans_0001152-1].

**Citations detected**: `medquad_ans_0000522-1__dup1`, `medquad_ans_0001152-1`
**Supported citations**: `medquad_ans_0000522-1__dup1`, `medquad_ans_0001152-1`
**Unsupported citations**: _none_

## Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000645-2` (score=0.7331) - Melnick-Needles syndrome is a rare disorder; fewer than 100 cases have been reported worldwide.
2. `medquad_ans_0000645-3` (score=0.7082) - Mutations in the FLNA gene cause Melnick-Needles syndrome. The FLNA gene provides instructions for producing the protein filamin A, which helps build the network of protein filaments (cytoskeleton) that gives structure to cells and allow...
3. `medquad_ans_0000645-5` (score=0.7049) - These resources address the diagnosis or management of Melnick-Needles syndrome: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Melnick-Needles syndrome These resources from MedlinePlus offer ...
4. `medquad_ans_0000372-2` (score=0.6593) - Feingold syndrome appears to be a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0001084-3__dup1` (score=0.6575) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).

**Answer**:

The provided context does not explicitly state whether Melnick-Needles syndrome is inherited. However, it mentions that mutations in the FLNA gene cause the syndrome and that these mutations are described as "gain-of-function" [medquad_ans_0000645-3]. Mutations in a gene such as FLNA are typically associated with inherited conditions, but the context does not directly confirm inheritance patterns.

Since the context does not explicitly state the inheritance pattern of Melnick-Needles syndrome, and no source directly addresses inheritance, the information is insufficient to definitively answer whether it is inherited.

Answer: Insufficient context to determine if Melnick-Needles syndrome is inherited. [medquad_ans_0000645-3]

**Citations detected**: `medquad_ans_0000645-3`
**Supported citations**: `medquad_ans_0000645-3`
**Unsupported citations**: _none_

## Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Top docs**:

1. `medquad_ans_0006138-6` (score=0.6584) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlying conditions contri...
2. `medquad_ans_0006138-1` (score=0.6502) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of TBM: a congenital fo...
3. `medquad_ans_0006140-6` (score=0.6473) - How might tracheobronchopathia osteoplastica be treated? There is no specific treatment for tracheobronchopathia osteoplastica (TO). Recurrent infections and collapse of the lung are treated conventionally. Inhaled corticosteroids may ha...
4. `medquad_ans_0006138-2` (score=0.6405) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or ...
5. `medquad_ans_0006138-5` (score=0.6367) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing such as CT scan and b...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) that is usually seen in adults [medquad_ans_0006138-1]. Most cases of primary TBM are caused by genetic conditions that weaken the airway walls, while secondary TBM often results from trauma, chronic inflammation, or prolonged compression of the airways [medquad_ans_0006138-1]. The condition is typically progressive and may lead to symptoms such as shortness of breath, cough, sputum retention

**Citations detected**: `medquad_ans_0006138-1`
**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

## Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.6682) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, ...
2. `medquad_ans_0000062-2` (score=0.6465) - Ankylosing spondylitis is part of a group of related diseases known as spondyloarthropathies. In the United States, spondyloarthropathies affect 3.5 to 13 per 1,000 people.
3. `medquad_ans_0000676-2__dup1` (score=0.6257) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0000531-1__dup2` (score=0.6241) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The two halves of the man...
5. `medquad_ans_0000086-2` (score=0.6199) - Auriculo-condylar syndrome appears to be a rare disorder. More than two dozen affected individuals have been described in the medical literature.

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Citations detected**: `medquad_ans_0006030-1`
**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

## Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000554-2` (score=0.7092) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
2. `medquad_ans_0000554-4` (score=0.6657) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one...
3. `medquad_ans_0000554-5` (score=0.6610) - These resources address the diagnosis or management of Kallmann syndrome: - Gene Review: Gene Review: Isolated Gonadotropin-Releasing Hormone (GnRH) Deficiency - Genetic Testing Registry: Hypogonadism with anosmia - Genetic Testing Regis...
4. `medquad_ans_0003339-1` (score=0.6606) - Kallmann syndrome (KS) is a condition characterized primarily by hypogonadotropic hypogonadism (HH) and absent or diminished sense of smell (anosmia or hyposmia, respectively). HH is present from birth and is due to deficiency of gonadot...
5. `medquad_ans_0000554-1` (score=0.6548) - Kallmann syndrome is a condition characterized by delayed or absent puberty and an impaired sense of smell. This disorder is a form of hypogonadotropic hypogonadism (HH), which is a condition affecting the production of hormones that dir...

**Answer**:

Yes, Kallmann syndrome can be inherited, but the pattern of inheritance varies depending on the specific form of the disorder. Kallmann syndrome 1, caused by mutations in the ANOS1 gene, follows an X-linked recessive pattern of inheritance, meaning males are more frequently affected than females, and affected individuals may inherit the mutation from their mothers who are carriers [medquad_ans_0000554-4]. Other forms of Kallmann syndrome can be inherited in an autosomal dominant or autosomal recessive pattern, where one or two copies of the altered gene, respectively, are sufficient to cause the disorder [medquad_ans_0000554-4]. In some cases, the condition results from new mutations with no family history of the disorder [medquad_ans_0000554-4]. Therefore, while not all cases are inherited, inheritance patterns vary and include X-linked

**Citations detected**: `medquad_ans_0000554-4`
**Supported citations**: `medquad_ans_0000554-4`
**Unsupported citations**: _none_

## Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Top docs**:

1. `medquad_ans_0005967-3` (score=0.6854) - How is syndactyly type 3 inherited? Syndactyly type 3 has been shown to be inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene is sufficient to cause the condition. When an indivi...
2. `medquad_ans_0004948-1` (score=0.6567) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no oth...
3. `medquad_ans_0005967-1` (score=0.6559) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bones of the fingers (di...
4. `medquad_ans_0005970-1` (score=0.6532) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0000970-1` (score=0.6530) - Tarsal-carpal coalition syndrome is a rare, inherited bone disorder that affects primarily the hands and feet. Several individual bones make up each wrist (carpal bones) and ankle (tarsal bones). In tarsal-carpal coalition syndrome, the ...

**Answer**:

Syndactyly type 3 is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases, only the soft tissue is fused, but in some cases, the bones of the fingers (distal phalanges) are also fused [medquad_ans_0005967-1]. This condition is inherited in an autosomal dominant manner and is caused by mutations in the GJA1 gene [medquad_ans_0005967-1].

**Citations detected**: `medquad_ans_0005967-1`
**Supported citations**: `medquad_ans_0005967-1`
**Unsupported citations**: _none_

## Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.6848) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tightness. There are tw...
2. `medquad_ans_0000021-4__dup2` (score=0.6741) - Bronchitis is a very common condition. Millions of cases occur every year. Elderly people, infants, and young children are at higher risk for acute bronchitis than people in other age groups. People of all ages can develop chronic bronch...
3. `medquad_ans_0000186-2` (score=0.6496) - Chronic granulomatous disease is estimated to occur in 1 in 200,000 to 250,000 people worldwide.
4. `medquad_ans_0000021-7` (score=0.6451) - The main goals of treating acute and chronic bronchitis are to relieve symptoms and make breathing easier. If you have acute bronchitis, your doctor may recommend rest, plenty of fluids, and aspirin (for adults) or acetaminophen to treat...
5. `medquad_ans_0000021-3__dup2` (score=0.6444) - Acute Bronchitis Infections or lung irritants cause acute bronchitis. The same viruses that cause colds and the flu are the most common cause of acute bronchitis. Sometimes bacteria can cause the condition. Certain substances can irritat...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs. It causes a cough that often brings up mucus and can also lead to shortness of breath, wheezing, a low fever, and chest tightness. It is one type of chronic obstructive pulmonary disease (COPD). The inflamed bronchial tubes produce excessive mucus, leading to persistent coughing and difficulty breathing. Cigarette smoking is the most common cause, and long-term exposure to air pollution, fumes, or dust also increases the risk. Chronic bronchitis is a long-term condition that recurs or never fully resolves. Women are more than twice as likely as men to be diagnosed with it, and it occurs more often in people over 45, especially smokers. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment focuses on relieving symptoms and includes medicines to

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Top docs**:

1. `medquad_ans_0000935-2` (score=0.6873) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.
2. `medquad_ans_0000173-2__dup2` (score=0.6691) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
3. `medquad_ans_0000201-4__dup1` (score=0.6641) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.
4. `medquad_ans_0000934-5` (score=0.6621) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - MedlinePlus Encyclopedi...
5. `medquad_ans_0000173-4__dup1` (score=0.6614) - Between 2003 and 2012, the NINDS piloted the Spinal Muscular Atrophy Project to expedite therapeutics development for this hereditary neurodegenerative disease. The Project was designed to accelerate the research process by identifying d...

**Answer**:

There is no cure for Spinal Muscular Atrophy. Treatment consists of managing the symptoms and preventing complications [medquad_ans_0000173-2__dup2]. Additional resources for diagnosis and management include genetic testing, drug therapy, genetic counseling, and palliative care [medquad_ans_0000934-5].

**Citations detected**: `medquad_ans_0000173-2__dup2`, `medquad_ans_0000934-5`
**Supported citations**: `medquad_ans_0000173-2__dup2`, `medquad_ans_0000934-5`
**Unsupported citations**: _none_

## Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Top docs**:

1. `medquad_ans_0000001_6-7` (score=0.6778) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood leukemia. - Children...
2. `medquad_ans_0000001_1-7` (score=0.6697) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - Chemotherapy with ste...
3. `medquad_ans_0000040-18` (score=0.6541) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certain features of the le...
4. `medquad_ans_0000040-19` (score=0.6344) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemia. Learn more about ...
5. `medquad_ans_0000040-10` (score=0.6295) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.

**Answer**:

The provided context does not contain information about treatments for adult acute lymphoblastic leukemia. It only discusses treatments for childhood acute lymphoblastic leukemia (ALL) [medquad_ans_0000001_6-7]. Therefore, based on the given context, the treatments for adult acute lymphoblastic leukemia cannot be determined.

**Citations detected**: `medquad_ans_0000001_6-7`
**Supported citations**: `medquad_ans_0000001_6-7`
**Unsupported citations**: _none_

## Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Top docs**:

1. `medquad_ans_0000133-3__dup1` (score=0.6893) - Generally, within 15 to 20 years after the appearance of the first symptoms, the person is confined to a wheelchair, and in later stages of the disease, individuals may become completely incapacitated. Friedreich's ataxia can shorten lif...
2. `medquad_ans_0002439-4` (score=0.6864) - How is Friedreich ataxia inherited? Friedreich ataxia is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. The parents of an aff...
3. `medquad_ans_0000388-2` (score=0.6818) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
4. `medquad_ans_0000133-4__dup1` (score=0.6811) - Friedreich's ataxia is caused by a mutation in the protein frataxin, which is involved in the function of mitochondriathe energy producing power plants of the cell. Frataxin controls important steps in mitochondrial iron metabolism and o...
5. `medquad_ans_0000388-5` (score=0.6746) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Friedreich ataxia 1 - M...

**Answer**:

Friedreich ataxia is caused by a mutation in the protein frataxin, which is involved in the function of mitochondria—the energy-producing power plants of the cell. Frataxin controls important steps in mitochondrial iron metabolism and overall cell iron stability [medquad_ans_0000133-4__dup1].

**Citations detected**: `medquad_ans_0000133-4__dup1`
**Supported citations**: `medquad_ans_0000133-4__dup1`
**Unsupported citations**: _none_

## Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Top docs**:

1. `medquad_ans_0000341-2` (score=0.7126) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with age. The incidence ...
2. `medquad_ans_0000341-1` (score=0.6889) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which causes the heartbeat to...
3. `medquad_ans_0000341-5` (score=0.6823) - These resources address the diagnosis or management of familial atrial fibrillation: - Genetic Testing Registry: Atrial fibrillation, familial, 1 - Genetic Testing Registry: Atrial fibrillation, familial, 2 - Genetic Testing Registry: At...
4. `medquad_ans_0002230-1` (score=0.6740) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular response in the heart's low...
5. `medquad_ans_0000341-4` (score=0.6661) - Familial atrial fibrillation appears to be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm, characterized by uncoordinated or erratic electrical activity in the heart's upper chambers (the atria), which causes a fast and irregular heartbeat (arrhythmia). This condition may be caused by changes (mutations) in various genes, some of which have not been identified, and it is most often inherited in an autosomal dominant manner, although autosomal recessive inheritance has been reported [medquad_ans_0002230-1]. The condition increases the risk of stroke and sudden death and may present with symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting; however, some individuals never experience associated health problems [medquad_ans_0000341-1]. The incidence of the familial form is unknown, but up to 30 percent of all people with atrial

**Citations detected**: `medquad_ans_0002230-1`, `medquad_ans_0000341-1`
**Supported citations**: `medquad_ans_0002230-1`, `medquad_ans_0000341-1`
**Unsupported citations**: _none_

## Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.6957) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary ...
2. `medquad_ans_0000166-2` (score=0.6538) - CADASIL is likely a rare condition; however, its prevalence is unknown.
3. `medquad_ans_0001440-1` (score=0.6507) - Colpocephaly is a congenital brain abnormality in which the occipital horns - the posterior or rear portion of the lateral ventricles (cavities) of the brain - are larger than normal because white matter in the posterior cerebrum has fai...
4. `medquad_ans_0003799-3` (score=0.6495) - What causes Marden-Walker syndrome? The underlying cause of Marden-Walker syndrome has not been clearly established. It appears to be a developmental disorder of the central nervous system and is likely to be the expression of various he...
5. `medquad_ans_0004627-2` (score=0.6489) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is available, the tabl...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia caused by mutations in the WTX gene and inherited in an X-linked dominant manner. It is characterized by a variety of features present at birth, including skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Additional symptoms may include developmental delay, hearing loss, heart defects, and ophthalmoplegia (paralysis of the muscles surrounding the eyes). The condition's severity and associated symptoms vary significantly among individuals, even within the same family. Treatment is based on the specific signs and symptoms present in each person [medquad_ans_0004627-1]. Specific signs and symptoms include abnormality of pelvic girdle bone morphology (90%), craniofacial hyperostosis (90%), rough bone trab

**Citations detected**: `medquad_ans_0004627-1`
**Supported citations**: `medquad_ans_0004627-1`
**Unsupported citations**: _none_

## Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Top docs**:

1. `medquad_ans_0004097-5` (score=0.7165) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is supported by a blood test...
2. `medquad_ans_0004097-1` (score=0.6923) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polymyositis. Some affect...
3. `medquad_ans_0004097-3` (score=0.6746) - What causes mixed connective tissue disease? The exact underlying cause of mixed connective tissue disease (MCTD) is currently unknown. It is an autoimmune disorder, which means the immune system mistakes normal, healthy cells for those ...
4. `medquad_ans_0004097-4` (score=0.6570) - Is mixed connective tissue disease inherited? The role of genetics in the onset of mixed connective tissue disease (MCTD) is still unclear. Some people with MCTD have family members who are also affected by the condition. This suggests t...
5. `medquad_ans_0004097-6` (score=0.6381) - How might mixed connective tissue disease be treated? There is currently no cure for mixed connective tissue disease (MCTD). However, treatments can help manage symptoms of the condition. For example, medications such as over-the-counter...

**Answer**:

The symptoms of mixed connective tissue disease (MCTD) may include Raynaud's phenomenon; arthritis; heart, lung, and skin abnormalities; kidney disease; muscle weakness; and dysfunction of the esophagus [medquad_ans_0004097-1].

**Citations detected**: `medquad_ans_0004097-1`
**Supported citations**: `medquad_ans_0004097-1`
**Unsupported citations**: _none_

## Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Top docs**:

1. `medquad_ans_0000004-21` (score=0.6691) - An early, accurate diagnosis of Alzheimer's disease helps people and their families plan for the future. It gives them time to discuss care options, find support, and make legal and financial arrangements while the person with Alzheimers...
2. `medquad_ans_0000003-1__dup4` (score=0.6644) - Now that your family member or friend has received a diagnosis of Alzheimers disease, its important to learn as much as you can about the disease and how to care for someone who has it. You may also want to know the right way to share th...
3. `medquad_ans_0000003-6__dup1` (score=0.6593) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the person will need more and...
4. `medquad_ans_0000098-2__dup1` (score=0.6573) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow the progression of t...
5. `medquad_ans_0000004-5__dup2` (score=0.6562) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate stages of the disease. ...

**Answer**:

The provided context does not mention specific treatments for Alzheimer's caregiving. It discusses the importance of early diagnosis, learning about the disease, understanding the stages of Alzheimer's, and coping with challenges, but it does not outline treatments or interventions for caregiving. Therefore, based on the given information, no answer can be provided regarding treatments for Alzheimer's caregiving. [medquad_ans_0000004-21][medquad_ans_0000003-1__dup4]

**Citations detected**: `medquad_ans_0000004-21`, `medquad_ans_0000003-1__dup4`
**Supported citations**: `medquad_ans_0000004-21`, `medquad_ans_0000003-1__dup4`
**Unsupported citations**: _none_

## Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Top docs**:

1. `medquad_ans_0000511-4` (score=0.6923) - Most cases of idiopathic inflammatory myopathy are sporadic, which means they occur in people with no history of the disorder in their family. However, several people with idiopathic inflammatory myopathy have had close relatives with au...
2. `medquad_ans_0000511-3` (score=0.6886) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identified variations in s...
3. `medquad_ans_0003141-1` (score=0.6738) - Idiopathic inflammatory myopathy refers to a group of conditions that affect the skeletal muscles (muscles used for movement). Although the condition can be diagnosed at any age, idiopathic inflammatory myopathy most commonly occurs in a...
4. `medquad_ans_0000511-5` (score=0.6513) - These resources address the diagnosis or management of idiopathic inflammatory myopathy: - Genetic Testing Registry: Idiopathic myopathy - Genetic Testing Registry: Inclusion body myositis - Johns Hopkins Myositis Center: Diagnosis - Joh...
5. `medquad_ans_0001800-3` (score=0.6467) - What causes dermatomyositis? The cause of this disorder is unknown. It is theorized that an autoimmune reaction (reactions caused by an immune response against the body's own tissues) or a viral infection of the skeletal muscle may cause...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations may influence the risk of developing the disorder by affecting how the immune system distinguishes between the body's own proteins and foreign invaders. Specific variations of certain HLA genes have been associated with an increased risk of idiopathic inflammatory myopathy. Researchers are also studying variations in other genes related to immune function to better understand their contribution to the condition's development [medquad_ans_0000511-3].

**Citations detected**: `medquad_ans_0000511-3`
**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

## Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0000855-2` (score=0.7269) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000855-5` (score=0.7201) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 deficiency - Genetic Te...
3. `medquad_ans_0005223-5` (score=0.7064) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for pyruvate dehydrogen...
4. `medquad_ans_0000854-5` (score=0.7059) - These resources address the diagnosis or management of pyruvate carboxylase deficiency: - Gene Review: Gene Review: Pyruvate Carboxylase Deficiency - Genetic Testing Registry: Pyruvate carboxylase deficiency These resources from MedlineP...
5. `medquad_ans_0005223-6` (score=0.7024) - How might pyruvate dehydrogenase deficiency be treated? Treatment of pyruvate dehydrogenase (PDH) deficiency rarely influences the course of the disease, but goals include stimulating the pyruvate dehydrogenase complex (PDHC), providing ...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Top docs**:

1. `medquad_ans_0006562-5` (score=0.7278) - How is 17q23.1q23.2 microdeletion syndrome diagnosed? The deletion can be identified by comparative genomic hybridization (CGH) microarray and fluorescence in situ hybridization (FISH).
2. `medquad_ans_0001086-2` (score=0.7183) - At least 500 cases of 22q13.3 deletion syndrome are known.
3. `medquad_ans_0006562-4` (score=0.7116) - Is 17q23.2q23.2 microdeletion syndrome inherited? Parental FISH testing in most of the reported cases confirmed a de novo origin, meaning that the deletion was new to the family.
4. `medquad_ans_0003415-3` (score=0.7043) - How is 17q21.31 microdeletion syndrome diagnosed? 17q21.31 microdeletion syndrome is diagnosed in individuals who have a deletion of 500,000 to 650,000 DNA building blocks (base pairs) at chromosome 17q21.31. The diagnosis can be made by...
5. `medquad_ans_0006562-1` (score=0.7001) - 17q23.1q23.2 microdeletion syndrome is a condition caused by a small deletion of genetic material from chromosome 17. The deletion occurs at a location encompassing bands 23.1 to 23.2 on the long (q) arm of the chromosome. People with 17...

**Answer**:

People with 17q23.1q23.2 microdeletion syndrome may have developmental delay, microcephaly, short stature, heart defects, and limb abnormalities [medquad_ans_0006562-1].

**Citations detected**: `medquad_ans_0006562-1`
**Supported citations**: `medquad_ans_0006562-1`
**Unsupported citations**: _none_

## Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0000635-3` (score=0.7020) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
2. `medquad_ans_0003860-3` (score=0.7003) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...
3. `medquad_ans_0000635-2` (score=0.6926) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of all cases of intell...
4. `medquad_ans_0003860-5` (score=0.6899) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the diagnosis.
5. `medquad_ans_0003860-4` (score=0.6825) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome, one of the two sex...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome [medquad_ans_0000635-3]. This duplication varies in size from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication [medquad_ans_0000635-3]. The extra copy leads to the production of excess MeCP2 protein, which is unable to properly regulate the expression of other genes in the brain, resulting in abnormal neuronal function and irregular brain activity [medquad_ans_0000635-3]. [medquad_ans_0003860-3]

**Citations detected**: `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Supported citations**: `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Unsupported citations**: _none_

## Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Top docs**:

1. `medquad_ans_0000759-2` (score=0.6961) - Autosomal dominant osteopetrosis is the most common form of the disorder, affecting about 1 in 20,000 people. Autosomal recessive osteopetrosis is rarer, occurring in an estimated 1 in 250,000 people. Other forms of osteopetrosis are ver...
2. `medquad_ans_0004638-1` (score=0.6926) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of inheritance: autosom...
3. `medquad_ans_0004640-1` (score=0.6926) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of inheritance: autosom...
4. `medquad_ans_0004635-1` (score=0.6926) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of inheritance: autosom...
5. `medquad_ans_0004633-1` (score=0.6926) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of inheritance: autosom...

**Answer**:

The provided context does not specify the symptoms of autosomal dominant type 2 osteopetrosis. While it mentions that osteopetrosis is characterized by abnormally dense bones prone to breakage and that types are distinguished by inheritance pattern and severity of symptoms, no details are given about the specific symptoms of autosomal dominant type 2 [medquad_ans_0000759-2][medquad_ans_0004638-1][medquad_ans_0004640-1][medquad_ans_0004635-1][medquad_ans_0004633-1]. Therefore, the symptoms cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000759-2`, `medquad_ans_0004638-1`, `medquad_ans_0004640-1`, `medquad_ans_0004635-1`, `medquad_ans_0004633-1`
**Supported citations**: `medquad_ans_0000759-2`, `medquad_ans_0004638-1`, `medquad_ans_0004640-1`, `medquad_ans_0004635-1`, `medquad_ans_0004633-1`
**Unsupported citations**: _none_

## Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Top docs**:

1. `medquad_ans_0000175-4__dup1` (score=0.6393) - The NINDS supports research on neuromuscular disorders such as LEMS with the ultimate goal of finding ways to treat, prevent, and cure them.
2. `medquad_ans_0003459-3` (score=0.6382) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases where Lambert Eaton ...
3. `medquad_ans_0003459-1` (score=0.6376) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when antibodies interfer...
4. `medquad_ans_0000201-4__dup1` (score=0.6373) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.
5. `medquad_ans_0000697-2` (score=0.6300) - Myasthenia gravis affects about 20 per 100,000 people worldwide. The prevalence has been increasing in recent decades, which likely results from earlier diagnosis and better treatments leading to longer lifespans for affected individuals.

**Answer**:

The treatments for Lambert Eaton myasthenic syndrome (LEMS) involve addressing the underlying disorder or cancer, as treatment of the underlying condition is the first priority [medquad_ans_0003459-1]. In cases associated with cancer, such as small cell lung cancer, treatment of the cancer is essential. However, specific pharmacological treatments for LEMS are not detailed in the provided context. [medquad_ans_0003459-3] and [medquad_ans_0003459-1] indicate that management focuses on the root cause rather than direct symptom treatment.

**Citations detected**: `medquad_ans_0003459-1`, `medquad_ans_0003459-3`
**Supported citations**: `medquad_ans_0003459-1`, `medquad_ans_0003459-3`
**Unsupported citations**: _none_

## Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Top docs**:

1. `medquad_ans_0000065-2` (score=0.7410) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
2. `medquad_ans_0000065-5` (score=0.7027) - These resources address the diagnosis or management of arginase deficiency: - Baby's First Test - Gene Review: Gene Review: Arginase Deficiency - Gene Review: Gene Review: Urea Cycle Disorders Overview - Genetic Testing Registry: Arginas...
3. `medquad_ans_0000445-3__dup1` (score=0.6974) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). This may be accomplis...
4. `medquad_ans_0000445-1__dup1` (score=0.6947) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of arginine in the blo...
5. `medquad_ans_0000065-3` (score=0.6904) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This cycle processes exce...

**Answer**:

The symptoms of arginase deficiency may include growth deficiency, spasticity (abnormal tensing of the muscles), developmental delay, loss of developmental milestones, intellectual disability, seizures, and microcephaly [medquad_ans_0000445-1__dup1].

**Citations detected**: `medquad_ans_0000445-1__dup1`
**Supported citations**: `medquad_ans_0000445-1__dup1`
**Unsupported citations**: _none_

## Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.7162) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but are still small so the...
2. `medquad_ans_0006376-3` (score=0.7071) - What causes Von Hippel-Lindau disease? Von Hippel-Lindau (VHL) disease is caused by a mutation in the VHL gene. This gene is a tumor suppressor gene, which helps to control cell growth. Mutations in the VHL gene lead to a lack of regulat...
3. `medquad_ans_0001020-3` (score=0.6899) - Mutations in the VHL gene cause von Hippel-Lindau syndrome. The VHL gene is a tumor suppressor gene, which means it keeps cells from growing and dividing too rapidly or in an uncontrolled way. Mutations in this gene prevent production of...
4. `medquad_ans_0001020-5` (score=0.6815) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau syndrome - Genomics Ed...
5. `medquad_ans_0001020-2` (score=0.6779) - The incidence of von Hippel-Lindau syndrome is estimated to be 1 in 36,000 individuals.

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Citations detected**: `medquad_ans_0006376-6`
**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

## Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7125) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-5` (score=0.6914) - These resources address the diagnosis or management of Noonan syndrome: - Gene Review: Gene Review: Noonan Syndrome - Genetic Testing Registry: Noonan syndrome - Genetic Testing Registry: Noonan syndrome 1 - Genetic Testing Registry: Noo...
3. `medquad_ans_0004450-3` (score=0.6873) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a pers...
4. `medquad_ans_0000738-3` (score=0.6639) - Mutations in the PTPN11, SOS1, RAF1, KRAS, NRAS and BRAF genes cause Noonan syndrome. Most cases of Noonan syndrome result from mutations in one of three genes, PTPN11, SOS1, or RAF1. PTPN11 gene mutations account for approximately 50 pe...
5. `medquad_ans_0004453-3` (score=0.6488) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...

**Answer**:

The provided context does not contain information about "Noonan syndrome 3." Therefore, it is not possible to answer the question based on the given text. [doc_id] not available

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Top docs**:

1. `medquad_ans_0005865-1` (score=0.6801) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, to the shoulder bein...
2. `medquad_ans_0005865-2` (score=0.6643) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may not have noticeable...
3. `medquad_ans_0005865-3` (score=0.6616) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may have physical therapy ...
4. `medquad_ans_0003402-2` (score=0.6516) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenital). The 3 most commo...
5. `medquad_ans_0000943-2` (score=0.6339) - Spondylocarpotarsal synostosis syndrome is a rare disorder; its prevalence is unknown. At least 25 affected individuals have been identified.

**Answer**:

The symptoms of Sprengel deformity include a lump in the back of the base of the neck, limited movement in the shoulder or arm, and abnormality of the shoulder (seen in approximately 90% of patients). Other associated symptoms may include underdeveloped or incomplete muscles in the surrounding area, bone, cartilage, or fibrous tissue between the shoulder blade and spinal bones, scoliosis, Klippel-Feil syndrome, limb length discrepancy, hemivertebrae, cervical rib abnormalities, abnormalities of the collarbone, abnormalities of the chest organs (such as the liver or heart being displaced), and spina bifida occulta or cleft palate (in approximately 7.5% of cases) [medquad_ans_0005865-1][medquad_ans_0005865-2].

**Citations detected**: `medquad_ans_0005865-1`, `medquad_ans_0005865-2`
**Supported citations**: `medquad_ans_0005865-1`, `medquad_ans_0005865-2`
**Unsupported citations**: _none_

## Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0005415-3` (score=0.7061) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell must have a mutation...
2. `medquad_ans_0000882-4` (score=0.7042) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an ind...
3. `medquad_ans_0000882-5` (score=0.6816) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Registry: Robinow syndro...
4. `medquad_ans_0005415-4` (score=0.6810) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of Robinow syndrome have...
5. `medquad_ans_0000538-4` (score=0.6704) - Isolated Pierre Robin sequence is usually not inherited. It typically results from new genetic changes and occurs in people with no history of the disorder in their family. When the condition is inherited, it follows an autosomal dominan...

**Answer**:

The genetic changes related to Robinow syndrome involve mutations in the responsible genes, which can be inherited in an autosomal recessive or autosomal dominant manner. In autosomal recessive inheritance, both copies of the gene in each cell must have a mutation for a person to be affected. In autosomal dominant inheritance, having only one mutated copy of the gene in each cell is sufficient to cause the condition [medquad_ans_0005415-3]. The specific gene involved in some cases is ROR2, and mutations in this gene are associated with ROR2-related Robinow syndrome [medquad_ans_0000882-4]. However, not all individuals with Robinow syndrome have mutations in the known genes [medquad_ans_0005415-4].

**Citations detected**: `medquad_ans_0005415-3`, `medquad_ans_0000882-4`, `medquad_ans_0005415-4`
**Supported citations**: `medquad_ans_0005415-3`, `medquad_ans_0000882-4`, `medquad_ans_0005415-4`
**Unsupported citations**: _none_

## Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Top docs**:

1. `medquad_ans_0000180-9` (score=0.6627) - - Kidney failure means your kidneys no longer work well enough to do their job. - Learning about treatments for kidney failure will help you choose the one that best fits your lifestyle. - Many people with kidney failure continue to work...
2. `medquad_ans_0000058-11__dup1` (score=0.6596) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
3. `medquad_ans_0000155-10` (score=0.6565) - Conservative management for kidney failure is the choice to say no to or stop dialysis treatments. For many people, dialysis not only extends life, it also improves the quality of life. For others who have serious conditions in addition ...
4. `medquad_ans_0000179-13` (score=0.6537) - - Kidney failure means your kidneys no longer filter your blood well enough to keep you healthy. - The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a kidney transplant - conservative management - Hemodialysis ...
5. `medquad_ans_0000038-5__dup2` (score=0.6509) - Kidney disease can get worse over time, and may lead to kidney failure. Kidney failure means advanced kidney damage with less than 15% normal function. Most people with kidney failure have symptoms from the build up of waste products and...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, a specific answer cannot be generated based on the given context. [doc_id] is not applicable.

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.7362) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000199-1__dup1` (score=0.7222) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty materials (lipids) ac...
3. `medquad_ans_0000676-2` (score=0.6867) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000675-2` (score=0.6854) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000674-2` (score=0.6824) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Citations detected**: `medquad_ans_0000677-2`
**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

## Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.7025) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin con...
2. `medquad_ans_0000120-11` (score=0.6928) - People can ask their health care providers about surgical techniques that minimize scar tissue. Scientists have not yet found a way to prevent short bowel syndrome that is present at birth, as its cause is unknown.
3. `medquad_ans_0000193-9` (score=0.6845) - - Cirrhosis is scarring of the liver. Scar tissue replaces healthy liver tissue. - Some common causes of cirrhosis include heavy alcohol use, hepatitis infections, and nonalcoholic fatty liver disease. - In the early stages of cirrhosis,...
4. `medquad_ans_0000193-1__dup4` (score=0.6762) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes proteins - helps figh...
5. `medquad_ans_0000193-6` (score=0.6740) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need to know about Liver T...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Citations detected**: `medquad_ans_0000795-1__dup1`
**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

## Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Top docs**:

1. `medquad_ans_0001045-4` (score=0.6744) - X-linked adrenoleukodystrophy is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males (who...
2. `medquad_ans_0001045-5` (score=0.6462) - These resources address the diagnosis or management of X-linked adrenoleukodystrophy: - Gene Review: Gene Review: X-Linked Adrenoleukodystrophy - Genetic Testing Registry: Adrenoleukodystrophy - Genomics Education Programme (UK) - Medlin...
3. `medquad_ans_0000784-5` (score=0.6348) - These resources address the diagnosis or management of peroxisomal acyl-CoA oxidase deficiency: - Gene Review: Gene Review: Leukodystrophy Overview - Genetic Testing Registry: Pseudoneonatal adrenoleukodystrophy These resources from Medl...
4. `medquad_ans_0001045-2` (score=0.6296) - The prevalence of X-linked adrenoleukodystrophy is 1 in 20,000 to 50,000 individuals worldwide. This condition occurs with a similar frequency in all populations.
5. `medquad_ans_0003566-1` (score=0.6221) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to defects of the myelin, ...

**Answer**:

The genetic changes related to X-linked adrenoleukodystrophy are mutations in the ABCD1 gene, which is located on the X chromosome. In males, one altered copy of the ABCD1 gene in each cell is sufficient to cause the disorder. Females, who have two copies of the X chromosome, usually do not develop features of the disorder with only one altered copy, although some may experience health problems associated with it [medquad_ans_0001045-4].

**Citations detected**: `medquad_ans_0001045-4`
**Supported citations**: `medquad_ans_0001045-4`
**Unsupported citations**: _none_

## Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.7325) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of European descent, where it...
2. `medquad_ans_0000988-4` (score=0.7128) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ra...
3. `medquad_ans_0000988-1` (score=0.7065) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most frequently occur in the...
4. `medquad_ans_0000027-1__dup3` (score=0.6868) - Amyloidosis occurs when abnormal proteins called amyloids build up and form deposits. The deposits can collect in organs such as the kidney and heart. This can cause the organs to become stiff and unable to work the way they should. Ther...
5. `medquad_ans_0000136-1__dup4` (score=0.6861) - Amyloidosis is a rare disease that occurs when amyloid proteins are deposited in tissues and organs. Amyloid proteins are abnormal proteins that the body cannot break down and recycle, as it does with normal proteins. When amyloid protei...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Citations detected**: `medquad_ans_0000988-2`
**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

## Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.6468) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the information is available, the ...
2. `medquad_ans_0000277-2` (score=0.5746) - Dentinogenesis imperfecta affects an estimated 1 in 6,000 to 8,000 people.
3. `medquad_ans_0000659-2` (score=0.5627) - Miller-Dieker syndrome appears to be a rare disorder, although its prevalence is unknown.
4. `medquad_ans_0002392-1` (score=0.5577) - Florid cemento-osseous dysplasia is characterized by lesions in the upper and/or lower jaw that occur when normal bone is replaced with a mix of connective tissue and abnormal bone. It tends to affect middle aged women, particularly wome...
5. `medquad_ans_0001013-2` (score=0.5575) - Van der Woude syndrome is believed to occur in 1 in 35,000 to 1 in 100,000 people, based on data from Europe and Asia. Van der Woude syndrome is the most common cause of cleft lip and palate resulting from variations in a single gene, an...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include features listed by the Human Phenotype Ontology (HPO), which compiles data on how often a symptom occurs in individuals with this condition. The frequency of these symptoms is typically estimated as a percentage or as a fraction (number of patients with the symptom out of the total number examined). For example, a frequency of 25/25 indicates that all patients in a study had the symptom. However, specific symptoms such as dental abnormalities or palatal invaginations are not detailed in the provided context. The information available confirms that the HPO has collected data on these signs and symptoms, with frequencies derived from sources like Orphanet, but does not list the exact symptoms [medquad_ans_0001780-1].

**Citations detected**: `medquad_ans_0001780-1`
**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

## Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.7003) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0004956-1` (score=0.6796) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associated with the condition...
3. `medquad_ans_0000720-1__dup1` (score=0.6778) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the entire surface of the br...
4. `medquad_ans_0000782-5` (score=0.6671) - These resources address the diagnosis or management of periventricular heterotopia: - Gene Review: Gene Review: FLNA-Related Periventricular Nodular Heterotopia - Genetic Testing Registry: Heterotopia, periventricular, associated with ch...
5. `medquad_ans_0004818-1` (score=0.6668) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected people typically develo...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Citations detected**: `medquad_ans_0000782-2`
**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

## Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Top docs**:

1. `medquad_ans_0003462-3` (score=0.6897) - How is Landau-Kleffner syndrome (LKS) diagnosed? LKS is diagnosed based on clinical features and the results of an electroencephalogram (EEG), a recording of the electric activity of the brain. All LKS children have abnormal electrical b...
2. `medquad_ans_0000179-4__dup1` (score=0.6709) - The NINDS conducts and supports a broad program of basic and clinical research on epilepsy including Lennox-Gastaut syndrome. These studies are aimed at finding the causes of these disorders, improving the diagnosis, and developing new m...
3. `medquad_ans_0000004-1__dup2` (score=0.6694) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encephalogram (EEG). LKS af...
4. `medquad_ans_0003462-1` (score=0.6667) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encephalogram (EEG). The di...
5. `medquad_ans_0000108-4__dup1` (score=0.6566) - The NINDS conducts and supports a broad program of basic and clinical research on all types of epilepsy, including Dravet syndrome. Study of the genetic defects responsible for Dravet syndrome and related disorders is expected to lead to...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for Landau-Kleffner Syndrome. While the NINDS conducts research on epilepsy, including Lennox-Gastaut syndrome, there is no information about research or clinical trials specifically for Landau-Kleffner Syndrome [medquad_ans_0000179-4__dup1]. Therefore, based on the given context, the research or clinical trials for LKS are not specified.

**Citations detected**: `medquad_ans_0000179-4__dup1`
**Supported citations**: `medquad_ans_0000179-4__dup1`
**Unsupported citations**: _none_

## Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Top docs**:

1. `medquad_ans_0000306-2` (score=0.7040) - Duchenne and Becker muscular dystrophies together affect 1 in 3,500 to 5,000 newborn males worldwide. Between 400 and 600 boys in the United States are born with these conditions each year.
2. `medquad_ans_0000395-2` (score=0.7006) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dystrophy has an estima...
3. `medquad_ans_0000631-1__dup1` (score=0.6883) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. The different types ...
4. `medquad_ans_0000294-2` (score=0.6880) - DMD-associated dilated cardiomyopathy appears to be an uncommon condition, although its prevalence is unknown.
5. `medquad_ans_0000602-2` (score=0.6861) - It is difficult to determine the prevalence of limb-girdle muscular dystrophy because its features vary and overlap with those of other muscle disorders. Prevalence estimates range from 1 in 14,500 to 1 in 123,000 individuals.

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases that cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood, while others may not appear until middle age or later. The different types can vary in whom they affect, which muscles they affect, and what the symptoms are. All forms of MD grow worse as the person's muscles get weaker, and most people with MD eventually lose the ability to walk. There is no cure for muscular dystrophy; treatments can help with symptoms and prevent complications, including physical and speech therapy, orthopedic devices, surgery, and medications. Some cases are mild and worsen slowly, while others are disabling and severe [medquad_ans_0000631-1__dup1].

**Citations detected**: `medquad_ans_0000631-1__dup1`
**Supported citations**: `medquad_ans_0000631-1__dup1`
**Unsupported citations**: _none_

## Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.7083) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no oth...
2. `medquad_ans_0003728-1` (score=0.6750) - Macrodactyly of the hand is a rare condition in which a person's fingers are abnormally large due to the overgrowth of the underlying bone and soft tissue. This condition is congenital, meaning that babies are born with it. Although babi...
3. `medquad_ans_0005967-3` (score=0.6686) - How is syndactyly type 3 inherited? Syndactyly type 3 has been shown to be inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene is sufficient to cause the condition. When an indivi...
4. `medquad_ans_0005970-1` (score=0.6583) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0001689-1` (score=0.6555) - What are the signs and symptoms of Curry Jones syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Curry Jones syndrome. If the information is available, the table below includes how often the sym...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no other physical anomalies or intellectual impairment. However, it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits. Surgical management depends greatly on the complexity of the deformity [1] [2].

**Citations detected**: `1`, `2`
**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

## Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Top docs**:

1. `medquad_ans_0004934-5` (score=0.6731) - When is Poland syndrome typically first diagnosed? The severity of Poland syndrome differs from person to person. As a result it is not often diagnosed or reported. Sometimes, a person does not realize they have the condition until puber...
2. `medquad_ans_0000758-1` (score=0.6692) - Osteoglophonic dysplasia is a condition characterized by abnormal bone growth that leads to severe head and face (craniofacial) abnormalities, dwarfism, and other features. The term osteoglophonic refers to the bones (osteo-) having dist...
3. `medquad_ans_0000176-1` (score=0.6640) - Cherubism is a disorder characterized by abnormal bone tissue in the lower part of the face. Beginning in early childhood, both the lower jaw (the mandible) and the upper jaw (the maxilla) become enlarged as bone is replaced with painles...
4. `medquad_ans_0002491-1` (score=0.6604) - Gardner-Diamond syndrome (GDS) is a rare condition characterized by episodes of unexplained, painful bruising that mostly occurs on the arms, legs, and/or face. It is most common in Caucasian women who have mental illness or emotional st...
5. `medquad_ans_0002524-1` (score=0.6604) - Genoa syndrome is a rare condition that primarily affects the brain and skull. Babies with this condition are generally born with semilobar holoprosencephaly, a disorder caused by failure of the developing brain to sufficiently divide in...

**Answer**:

The provided context does not contain any information about Geniospasm or its symptoms. Therefore, it is not possible to answer the question based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000512-4` (score=0.6725) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance. Autosomal dominant ...
2. `medquad_ans_0000512-3` (score=0.6464) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of developing idiopathic pul...
3. `medquad_ans_0000512-5` (score=0.6440) - These resources address the diagnosis or management of idiopathic pulmonary fibrosis: - Gene Review: Gene Review: Pulmonary Fibrosis, Familial - Genetic Testing Registry: Idiopathic fibrosing alveolitis, chronic form These resources from...
4. `medquad_ans_0000512-2` (score=0.6349) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Familial pulmonary fibros...
5. `medquad_ans_0000037-3` (score=0.6069) - The cause of allergic asthma is complex. It is likely that a combination of multiple genetic and environmental factors contribute to development of the condition. Doctors believe genes are involved because having a family member with all...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in genes known as TERC and TERT, which have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. However, it is not well understood how defects in telomerase are associated with the lung damage characteristic of the disease. Additionally, changes in several genes have been suggested as risk factors, though most account for only a small proportion of cases [medquad_ans_0000512-3].

**Citations detected**: `medquad_ans_0000512-3`
**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

## Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000813-4` (score=0.7334) - Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected person inherits the chr...
2. `medquad_ans_0000813-2` (score=0.7281) - Potocki-Shaffer syndrome is a rare condition, although its prevalence is unknown. Fewer than 100 cases have been reported in the scientific literature.
3. `medquad_ans_0001084-3__dup1` (score=0.7074) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0002717-4` (score=0.7043) - How is Hallermann-Streiff syndrome inherited? The majority of cases of Hallermann-Streiff syndrome appear to be sporadic (occurring in individuals with no history of the condition in the family). There have been reports of affected indiv...
5. `medquad_ans_0005603-3` (score=0.7022) - How is SHORT syndrome inherited? SHORT syndrome is inherited in an autosomal dominant pattern. For conditions with autosomal dominant inheritance, one abnormal copy of the causative gene in each cell is sufficient to cause signs and symp...

**Answer**:

Yes, Potocki-Shaffer syndrome is inherited in an autosomal dominant pattern. A deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. This can occur when an affected person inherits the chromosome with a deleted segment from an affected parent, or when the deletion occurs during the formation of reproductive cells (eggs and sperm) or in early fetal development, resulting in individuals with no family history of the disorder [medquad_ans_0000813-4].

**Citations detected**: `medquad_ans_0000813-4`
**Supported citations**: `medquad_ans_0000813-4`
**Unsupported citations**: _none_

## Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Top docs**:

1. `medquad_ans_0000925-1__dup1` (score=0.7559) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about your baby's health, ca...
2. `medquad_ans_0000897-1__dup1` (score=0.7430) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call your health care provi...
3. `medquad_ans_0000691-1__dup1` (score=0.7395) - If you're a parent, you get plenty of suggestions on how to raise your child. From experts to other parents, people are always ready to offer advice. Parenting tips, parents' survival guides, dos, don'ts, shoulds and shouldn'ts - new one...
4. `medquad_ans_0000076-1__dup3` (score=0.7390) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developing properly. If ther...
5. `medquad_ans_0000177-1__dup2` (score=0.7275) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may choose in-home care, w...

**Answer**:

Yes, the context mentions that there are different types of child care options, including in-home care, where the caregiver comes to your home, or the child might go to the caregiver's home, and there are also child care centers. You need to choose the one that works for you [medquad_ans_0000177-1__dup2].

**Citations detected**: `medquad_ans_0000177-1__dup2`
**Supported citations**: `medquad_ans_0000177-1__dup2`
**Unsupported citations**: _none_

## Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.7567) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000046-4` (score=0.7263) - Alternating hemiplegia of childhood is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of alternating hemiplegia of childhood result from n...
3. `medquad_ans_0000016-3__dup1` (score=0.7140) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy, and balance and gai...
4. `medquad_ans_0000046-3` (score=0.7123) - Alternating hemiplegia of childhood is primarily caused by mutations in the ATP1A3 gene. Very rarely, a mutation in the ATP1A2 gene is involved in the condition. These genes provide instructions for making very similar proteins. They fun...
5. `medquad_ans_0000016-1__dup1` (score=0.7047) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or both sides of the body,...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Citations detected**: `medquad_ans_0000046-2`
**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

## Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Top docs**:

1. `medquad_ans_0000159-1__dup3` (score=0.6962) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 percent of people with...
2. `medquad_ans_0000186-3__dup1` (score=0.6857) - The prognosis for lupus varies widely depending on the organs involved and the intensity of the inflammatory reaction. The course of lupus is commonly chronic and relapsing, often with long periods of remission. Most individuals with lup...
3. `medquad_ans_0000159-3__dup2` (score=0.6615) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time or shortly after lu...
4. `medquad_ans_0003688-1` (score=0.6591) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, high blood pressure, ...
5. `medquad_ans_0000159-8` (score=0.6582) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed through urine and b...

**Answer**:

The symptoms of lupus may include high blood pressure, foamy urine, edema (swelling), usually in the legs, feet, or ankles and less often in the hands or face, joint pain or swelling, muscle pain, fever with no known cause, and red rashes, often on the face, which are also called butterfly rashes because of their shape [medquad_ans_0000159-3__dup2]. Additionally, blood in the urine and a foamy appearance to the urine are also associated with lupus nephritis, a complication of lupus [medquad_ans_0003688-1].

**Citations detected**: `medquad_ans_0000159-3__dup2`, `medquad_ans_0003688-1`
**Supported citations**: `medquad_ans_0000159-3__dup2`, `medquad_ans_0003688-1`
**Unsupported citations**: _none_

## Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Top docs**:

1. `medquad_ans_0000194-2` (score=0.7317) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.
2. `medquad_ans_0000194-3` (score=0.7029) - Cockayne syndrome can result from mutations in either the ERCC6 gene (also known as the CSB gene) or the ERCC8 gene (also known as the CSA gene). These genes provide instructions for making proteins that are involved in repairing damaged...
3. `medquad_ans_0000194-5` (score=0.6975) - These resources address the diagnosis or management of Cockayne syndrome: - Gene Review: Gene Review: Cockayne Syndrome - Genetic Testing Registry: Cockayne syndrome - Genetic Testing Registry: Cockayne syndrome type A - Genetic Testing ...
4. `medquad_ans_0001397-1` (score=0.6537) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
5. `medquad_ans_0001395-1` (score=0.6537) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...

**Answer**:

Cockayne syndrome is a rare condition that causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. It also includes failure to thrive in the newborn, microcephaly, and impaired nervous system development. Other symptoms may include hearing loss, tooth decay, and eye and bone abnormalities. The syndrome is caused by mutations in either the ERCC8 (CSA) or ERCC6 (CSB) genes and is inherited in an autosomal recessive pattern. There are three types: type 1 (type A), which is sometimes called classic or "moderate" Cockayne syndrome and is diagnosed during early childhood; type 2 (type B), referred to as the severe or "early-onset" type, presenting with growth and developmental abnormalities at birth; and type 3 (type C), which is a milder form. The typical lifespan for individuals with type 1 is ten to twenty

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Top docs**:

1. `medquad_ans_0000029-6__dup1` (score=0.7083) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead to sore or bleeding...
2. `medquad_ans_0000413-1__dup1` (score=0.7030) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the teeth. In the worst c...
3. `medquad_ans_0000029-8__dup1` (score=0.6963) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regular cleaning by a de...
4. `medquad_ans_0000029-11` (score=0.6903) - If left untreated, gum disease can lead to tooth loss. Gum disease is the leading cause of tooth loss in older adults.
5. `medquad_ans_0000029-9__dup1` (score=0.6843) - When gingivitis is not treated, it can advance to periodontitis (which means "inflammation around the tooth.") In periodontitis, gums pull away from the teeth and form "pockets" that become infected. The body's immune system fights the b...

**Answer**:

Gum (periodontal) disease is an infection of the tissues that hold your teeth in place. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the teeth. In gingivitis, the gums become red, swollen, and can bleed easily; this is a mild form of gum disease that can usually be reversed with daily brushing and flossing and regular cleanings by a dentist or dental hygienist. If left untreated, gingivitis can advance to periodontitis, where the gums pull away from the teeth and form infected pockets. The body's immune response to bacteria beneath the gum line leads to the breakdown of the bone and tissue that support the teeth. If not treated, this can result in destruction of the gums, bone, and connective tissue, leading to loose teeth and eventual tooth loss [medquad_ans_0000029-6\_\_dup1

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.7449) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000505-2` (score=0.7073) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
3. `medquad_ans_0003059-1` (score=0.6948) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles in the arms and legs....
4. `medquad_ans_0003059-5` (score=0.6844) - How is hypokalemic periodic paralysis diagnosed? The diagnosis of hypokalemic periodic paralysis is based on a history of episodes of paralysis and low levels of potassium in the blood during attacks (less than 0.9 to 3.0 mmol/L), but no...
5. `medquad_ans_0003000-1` (score=0.6817) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Citations detected**: `medquad_ans_0000494-2`
**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

## Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Top docs**:

1. `medquad_ans_0005212-3` (score=0.7574) - Is Pyle disease inherited? Pyle disease in inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one cop...
2. `medquad_ans_0000643-1__dup1` (score=0.6916) - Barrett esophagus is a condition in which the lining of the esophagus (the tube that carries food from the throat to the stomach) is replaced by tissue that is similar to the lining of the intestines. Although this change does not cause ...
3. `medquad_ans_0000103-10` (score=0.6835) - - Gastritis is a condition in which the stomach liningknown as the mucosais inflamed, or swollen. - Common causes of gastritis include Helicobacter pylori (H. pylori) infection, damage to the stomach lining, and an autoimmune response. -...
4. `medquad_ans_0000530-5__dup1` (score=0.6811) - How is autoimmune atrophic gastritis diagnosed? A diagnosis of autoimmune atrophic gastritis is generally not suspected until characteristic signs and symptoms are present. Additional testing can then be ordered to confirm the diagnosis....
5. `medquad_ans_0000739-2` (score=0.6795) - Norrie disease is a rare disorder; its exact incidence is unknown. It is not associated with any specific racial or ethnic group.

**Answer**:

Pyle disease is inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with Pyle disease each carry one copy of the mutated gene, but they often do not have any signs or symptoms of the condition. Some carriers (obligate heterozygotes) of Pyle disease show minor skeletal changes [medquad_ans_0005212-3].

**Citations detected**: `medquad_ans_0005212-3`
**Supported citations**: `medquad_ans_0005212-3`
**Unsupported citations**: _none_

## Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Top docs**:

1. `medquad_ans_0000256-1__dup3` (score=0.6752) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thrombosis can break loo...
2. `medquad_ans_0000051-4__dup2` (score=0.6712) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden) will do this. Horm...
3. `medquad_ans_0000456-1` (score=0.6682) - Hereditary antithrombin deficiency is a disorder of blood clotting. People with this condition are at higher than average risk for developing abnormal blood clots, particularly a type of clot that occurs in the deep veins of the legs. Th...
4. `medquad_ans_0000104-1__dup4` (score=0.6679) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent blood clots from diss...
5. `medquad_ans_0000051-3__dup2` (score=0.6677) - Blood clots can form in your body's deep veins if: A vein's inner lining is damaged. Injuries caused by physical, chemical, or biological factors can damage the veins. Such factors include surgery, serious injuries, inflammation, and imm...

**Answer**:

Deep vein thrombosis (DVT) is caused by a combination of factors that increase the likelihood of blood clots forming in deep veins. These include conditions or factors that make blood thicker or more likely to clot than normal, such as inherited blood disorders (e.g., factor V Leiden or hereditary antithrombin deficiency), hormone therapy or birth control pills, injury to a deep vein from surgery, broken bones, or trauma, slow blood flow due to lack of movement (e.g., after surgery, being ill and bedridden, or prolonged travel), pregnancy and the first 6 weeks postpartum, recent or ongoing cancer treatment, presence of a central venous catheter, older age (over 60), being overweight or obese, and smoking [medquad_ans_0000051-4__dup2]. Additionally, sitting still for long periods can increase the risk of DVT [medquad*ans*

**Citations detected**: `medquad_ans_0000051-4__dup2`
**Supported citations**: `medquad_ans_0000051-4__dup2`
**Unsupported citations**: _none_

## Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Top docs**:

1. `medquad_ans_0000661-2` (score=0.6957) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.
2. `medquad_ans_0004079-4` (score=0.6872) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Foundation (UMDF) provides...
3. `medquad_ans_0000661-5` (score=0.6857) - These resources address the diagnosis or management of mitochondrial complex III deficiency: - Gene Review: Gene Review: Mitochondrial Disorders Overview - Genetic Testing Registry: MITOCHONDRIAL COMPLEX III DEFICIENCY, NUCLEAR TYPE 6 - ...
4. `medquad_ans_0000178-3__dup2` (score=0.6747) - The prognosis for individuals with Leigh's disease is poor. Individuals who lack mitochondrial complex IV activity and those with pyruvate dehydrogenase deficiency tend to have the worst prognosis and die within a few years. Those with p...
5. `medquad_ans_0000692-2` (score=0.6505) - Multiple mitochondrial dysfunctions syndrome is a rare condition; its prevalence is unknown. It is one of several conditions classified as mitochondrial disorders, which affect an estimated 1 in 5,000 people worldwide.

**Answer**:

The provided context does not contain specific information about mitochondrial complex IV deficiency. While it mentions that individuals who lack mitochondrial complex IV activity tend to have the worst prognosis and die within a few years, no detailed description of mitochondrial complex IV deficiency itself is given [medquad_ans_0000178-3__dup2]. Therefore, based on the available context, the specific nature or characteristics of mitochondrial complex IV deficiency cannot be determined.

**Citations detected**: `medquad_ans_0000178-3__dup2`
**Supported citations**: `medquad_ans_0000178-3__dup2`
**Unsupported citations**: _none_

## Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Top docs**:

1. `medquad_ans_0000078-2` (score=0.7015) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
2. `medquad_ans_0000323-2` (score=0.6770) - Episodic ataxia is uncommon, affecting less than 1 in 100,000 people. Only types 1 and 2 have been identified in more than one family, and type 2 is by far the most common form of the condition.
3. `medquad_ans_0000078-1` (score=0.6761) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most affected people also ...
4. `medquad_ans_0000078-5` (score=0.6670) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 2 - Genetic Testing ...
5. `medquad_ans_0000495-1__dup1` (score=0.6622) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before age 5. The condition i...

**Answer**:

Ataxia with oculomotor apraxia is a rare condition characterized by progressive problems with movement, including difficulty coordinating movements (ataxia), which is often the first symptom. A hallmark feature is oculomotor apraxia, making it difficult to move the eyes side-to-side, requiring individuals to turn their head to see peripheral vision. The condition has multiple types, with types 1 and 2 being the most common. Type 1 typically presents with ataxia beginning around age 4, while type 2 begins around age 15. Both types share features such as involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy); however, chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Type 1 individuals often develop hand and foot wasting (atrophy), and nearly all

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Top docs**:

1. `medquad_ans_0000221-5` (score=0.6965) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with labyrinthine aplasia mic...
2. `medquad_ans_0000221-1` (score=0.6962) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that form the inner ear are...
3. `medquad_ans_0000248-1` (score=0.6622) - Craniofacial microsomia is a term used to describe a spectrum of abnormalities that primarily affect the development of the skull (cranium) and face before birth. Microsomia means abnormal smallness of body structures. Most people with c...
4. `medquad_ans_0001744-1` (score=0.6559) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine aplasia microtia and mic...
5. `medquad_ans_0004109-1` (score=0.6460) - Mondini dysplasia is a type of inner ear malformation that is present at birth (congenital). Individuals with Mondini dysplasia have one and a half coils of the cochlea instead of the normal two coils. It may occur in one ear (unilateral...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed in affected infants. In addition, people with LAMM syndrome often have abnormally small outer ears (microtia) with narrow ear canals. They can also have unusually small, widely spaced teeth (microdontia) [medquad_ans_000

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.6949) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in the gene, and occur ...
2. `medquad_ans_0000064-2` (score=0.6783) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.
3. `medquad_ans_0000064-3` (score=0.6575) - Mutations in the FGFR2 gene cause Apert syndrome. This gene produces a protein called fibroblast growth factor receptor 2. Among its multiple functions, this protein signals immature cells to become bone cells during embryonic developmen...
4. `medquad_ans_0000255-3` (score=0.6461) - Mutations in the FGFR2 gene cause Crouzon syndrome. This gene provides instructions for making a protein called fibroblast growth factor receptor 2. Among its multiple functions, this protein signals immature cells to become bone cells d...
5. `medquad_ans_0000064-5` (score=0.6435) - These resources address the diagnosis or management of Apert syndrome: - Gene Review: Gene Review: FGFR-Related Craniosynostosis Syndromes - Genetic Testing Registry: Acrocephalosyndactyly type I - MedlinePlus Encyclopedia: Apert syndrom...

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in people with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4]. Therefore, while the disorder follows an autosomal dominant pattern, it is most commonly not inherited from parents but arises from new mutations.

**Citations detected**: `medquad_ans_0000064-4`
**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

## Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.6312) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organisms are excreted in ...
2. `medquad_ans_0000341-1__dup3` (score=0.6268) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella burnetii usually devel...
3. `medquad_ans_0000341-3__dup2` (score=0.6230) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact the Rickettsial Zoon...
4. `medquad_ans_0000361-2` (score=0.6166) - Familial Mediterranean fever primarily affects populations originating in the Mediterranean region, particularly people of Armenian, Arab, Turkish, or Jewish ancestry. The disorder affects 1 in 200 to 1,000 people in these populations. I...
5. `medquad_ans_0005360-5` (score=0.6123) - How is rheumatic fever diagnosed? A diagnosis of rheumatic fever is usually based on the following: Characteristic signs and symptoms identified by physical examination and/or specialized testing such as a blood test, chest X-ray and ech...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. The acute form is treated with antibiotics, while treatment for the chronic form depends on the symptoms

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Top docs**:

1. `medquad_ans_0001152-2` (score=0.6713) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment for chILD is to rel...
2. `medquad_ans_0001152-1` (score=0.6503) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD decrease a child's ab...
3. `medquad_ans_0000038-7` (score=0.6260) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other diseases as the cause of ...
4. `medquad_ans_0000065-3__dup1` (score=0.6234) - Most children with the classic form of Farbers disease die by age 2, usually from lung disease. Children born with the most severe form of the disease usually die within 6 months, while individuals having a milder form of the disease may...
5. `medquad_ans_0000073-3__dup1` (score=0.6201) - COFS is a fatal disease. Most children do not live beyond five years.

**Answer**:

The provided context does not specify who is at risk for Childhood Interstitial Lung Disease (chILD). While it mentions factors such as exposure to lung irritants, autoimmune diseases, birth defects, lung infections, and radiation or chemotherapy treatment, it does not clearly identify specific risk groups or populations. Therefore, based on the given information, the risk factors for chILD cannot be determined. [medquad_ans_0001152-1] [medquad_ans_0000038-7]

**Citations detected**: `medquad_ans_0001152-1`, `medquad_ans_0000038-7`
**Supported citations**: `medquad_ans_0001152-1`, `medquad_ans_0000038-7`
**Unsupported citations**: _none_

## Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.6730) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in late childhood or early...
2. `medquad_ans_0000555-2` (score=0.6074) - In the United States and other Western countries, Kawasaki disease occurs in approximately 1 in 10,000 children under 5 each year. The condition is 10 to 20 times more common in East Asia, including Japan, Korea, and Taiwan.
3. `medquad_ans_0000083-8` (score=0.6012) - Kawasaki disease can't be prevented. However, most children who have the disease recoverusually within weeks of getting signs and symptoms. Further problems are rare.
4. `medquad_ans_0000083-4__dup2` (score=0.5940) - Kawasaki disease affects children of all races and ages and both genders. It occurs most often in children of Asian and Pacific Island descent. The disease is more likely to affect boys than girls. Most cases occur in children younger th...
5. `medquad_ans_0000136-5__dup1` (score=0.5905) - Vasculitis can affect people of all ages and races and both sexes. Some types of vasculitis seem to occur more often in people who: Have certain medical conditions, such as chronic hepatitis B or C infection Have certain autoimmune disea...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Citations detected**: `medquad_ans_0003334-1`
**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

## Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0001082-2` (score=0.7505) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyryl-CoA dehydrogenase ...
2. `medquad_ans_0006577-1` (score=0.7434) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particular amino acid called...
3. `medquad_ans_0001082-5` (score=0.7202) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from MedlinePlus offer i...
4. `medquad_ans_0001082-3` (score=0.7170) - Mutations in the ACADSB gene cause 2-methylbutyryl-CoA dehydrogenase deficiency. The ACADSB gene provides instructions for making an enzyme called 2-methylbutyryl-CoA dehydrogenase that helps process the amino acid isoleucine. Mutations ...
5. `medquad_ans_0001082-1` (score=0.7088) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as organic acids. Abno...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare metabolic disorder caused by mutations in the ACADSB gene, which result in a deficiency of the enzyme 2-methylbutyryl-CoA dehydrogenase. This enzyme is essential for the proper processing of the amino acid isoleucine. Without sufficient enzyme activity, isoleucine cannot be broken down effectively, leading to its accumulation in the body. This buildup can cause a range of health problems, from mild to severe and life-threatening, including poor feeding, lack of energy, vomiting, irritability, lethargy, and muscle weakness. Additionally, harmful organic acids such as 2-methylbutyrylglycine and related compounds may accumulate, contributing to toxicity and serious health complications. The disorder is more common among individuals of Hmong ancestry, with an incidence of 1 in 250 to 1 in 5

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Top docs**:

1. `medquad_ans_0001064-2` (score=0.6815) - The prevalence of X-linked spondyloepiphyseal dysplasia tarda is estimated to be 1 in 150,000 to 200,000 people worldwide.
2. `medquad_ans_0001064-5` (score=0.6738) - These resources address the diagnosis or management of X-linked spondyloepiphyseal dysplasia tarda: - Gene Review: Gene Review: X-Linked Spondyloepiphyseal Dysplasia Tarda - Genetic Testing Registry: Spondyloepiphyseal dysplasia tarda Th...
3. `medquad_ans_0000944-2` (score=0.6705) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0000947-1` (score=0.6699) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of the spine (spondylo-)...
5. `medquad_ans_0000947-5` (score=0.6659) - These resources address the diagnosis or management of spondyloepiphyseal dysplasia congenita: - Genetic Testing Registry: Spondyloepiphyseal dysplasia congenita - MedlinePlus Encyclopedia: Clubfoot - MedlinePlus Encyclopedia: Lordosis -...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include short stature from birth, a very short trunk and neck, shortened limbs, abnormal curvature of the spine (kyphoscoliosis and lordosis) that worsens during childhood, instability of the spinal bones in the neck increasing the risk of spinal cord damage, flattened vertebrae (platyspondyly), an abnormality of the hip joint causing the upper leg bones to turn inward (coxa vara), a foot deformity called clubfoot, a broad, barrel-shaped chest that can cause breathing problems, arthritis and decreased joint mobility that develop early in life, mild changes in facial features such as flattened cheekbones near the nose, an opening in the roof of the mouth (cleft palate) in some infants, severe nearsightedness (high myopia), and other eye problems that can impair vision. About one quarter of people have hearing loss [medquad

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Top docs**:

1. `medquad_ans_0000853-2` (score=0.7311) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
2. `medquad_ans_0000852-2` (score=0.7080) - Pyridoxal 5'-phosphate-dependent epilepsy is a rare condition; approximately 14 cases have been described in the scientific literature.
3. `medquad_ans_0005219-4` (score=0.6917) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of seizure are medically ...
4. `medquad_ans_0000852-3` (score=0.6835) - Mutations in the PNPO gene cause pyridoxal 5'-phosphate-dependent epilepsy. The PNPO gene provides instructions for producing an enzyme called pyridoxine 5'-phosphate oxidase. This enzyme is involved in the conversion (metabolism) of vit...
5. `medquad_ans_0005219-1` (score=0.6808) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. [medquad_ans_0005219-1]

**Citations detected**: `medquad_ans_0005219-1`
**Supported citations**: `medquad_ans_0005219-1`
**Unsupported citations**: _none_

## Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Top docs**:

1. `medquad_ans_0000004_6-5` (score=0.6216) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whether there are B symp...
2. `medquad_ans_0000004_6-2` (score=0.6164) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk f...
3. `medquad_ans_0000004_7-7` (score=0.6144) - Key Points - There are different types of treatment for children with non-Hodgkin lymphoma. - Children with non-Hodgkin lymphoma should have their treatment planned by a team of doctors who are experts in treating childhood cancer. - Som...
4. `medquad_ans_0000004_6-8` (score=0.6141) - Key Points - There are different types of treatment for children with Hodgkin lymphoma. - Children with Hodgkin lymphoma should have their treatment planned by a team of health care providers who are experts in treating childhood cancer....
5. `medquad_ans_0000004_1-2` (score=0.6013) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not havin...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma [medquad_ans_0000004_6-2]. Having a risk factor does not mean that a person will get cancer, and not having risk factors does not mean that a person will not get cancer [medquad_ans_0000004_6-2].

**Citations detected**: `medquad_ans_0000004_6-2`
**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

## Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Top docs**:

1. `medquad_ans_0001795-3` (score=0.6507) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, dapsone may cause ser...
2. `medquad_ans_0000095-2__dup3` (score=0.6468) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immune response to gluten...
3. `medquad_ans_0001795-1` (score=0.6407) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow onset of symptoms u...
4. `medquad_ans_0000095-1__dup4` (score=0.6386) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation may precede lesion f...
5. `medquad_ans_0000084-2` (score=0.6335) - Atopic dermatitis is a common disorder that affects 10 to 20 percent of children and 5 to 10 percent of adults.

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, commonly found on the knees, elbows, buttocks, and shoulder blades. It is frequently associated with the inability to digest gluten and is caused by the deposit of immunoglobulin A (IgA) in the skin, triggering an immunologic reaction that results in lesion formation. This condition is an external manifestation of an abnormal immune response to gluten, in which IgA antibodies form against the skin antigen epidermal transglutaminase. Family studies show that 5 percent of first-degree relatives of a person with DH will also have DH, and an additional 5 percent will have celiac disease. Other autoimmune diseases, such as hypothyroidism, are also associated with DH. [medquad_ans_0001795-1]medquad_ans_000009

**Citations detected**: `medquad_ans_0001795-1`
**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

## Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Top docs**:

1. `medquad_ans_0000709-2` (score=0.7119) - Myotonia congenita is estimated to affect 1 in 100,000 people worldwide. This condition is more common in northern Scandinavia, where it occurs in approximately 1 in 10,000 people.
2. `medquad_ans_0000771-2` (score=0.6818) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.
3. `medquad_ans_0000709-5` (score=0.6793) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Registry: Congenital myo...
4. `medquad_ans_0000210-3__dup1` (score=0.6786) - Myotonia is a chronic disorder. Symptoms may improve later in life.
5. `medquad_ans_0000040-2__dup1` (score=0.6637) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some succes...

**Answer**:

Most people with myotonia congenita do not require special treatments. Stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs such as phenytoin. Physical therapy and other rehabilitative therapies are also sometimes used to improve muscle function [medquad_ans_0000040-2__dup1].

**Citations detected**: `medquad_ans_0000040-2__dup1`
**Supported citations**: `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

## Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.7865) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0006449-5` (score=0.7538) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, whi...
3. `medquad_ans_0000276-4__dup1` (score=0.7527) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct and/or support rese...
4. `medquad_ans_0006449-3` (score=0.7490) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body. It also helps remo...
5. `medquad_ans_0006449-4` (score=0.7410) - Is Wilson disease inherited? Wilson disease is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. The parents of an affected pers...

**Answer**:

There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, which may include certain medications and/or dietary modifications. If treatment is not effective or if liver failure develops, a liver transplant may be necessary [medquad_ans_0006449-5].

**Citations detected**: `medquad_ans_0006449-5`
**Supported citations**: `medquad_ans_0006449-5`
**Unsupported citations**: _none_

## Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Top docs**:

1. `medquad_ans_0000179-4__dup1` (score=0.6635) - The NINDS conducts and supports a broad program of basic and clinical research on epilepsy including Lennox-Gastaut syndrome. These studies are aimed at finding the causes of these disorders, improving the diagnosis, and developing new m...
2. `medquad_ans_0000592-3` (score=0.6532) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are related to an existing...
3. `medquad_ans_0000592-2` (score=0.6493) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in males than in females.
4. `medquad_ans_0000592-4` (score=0.6463) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of some type of epilepsy. ...
5. `medquad_ans_0000108-4__dup1` (score=0.6416) - The NINDS conducts and supports a broad program of basic and clinical research on all types of epilepsy, including Dravet syndrome. Study of the genetic defects responsible for Dravet syndrome and related disorders is expected to lead to...

**Answer**:

The context does not provide information on specific genetic changes related to Lennox-Gastaut syndrome. It states that researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. Therefore, based on the provided information, no specific genetic changes are described.

**Citations detected**: `medquad_ans_0000592-3`
**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

## Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.7644) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-3` (score=0.7409) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step production (synthesi...
3. `medquad_ans_0000066-5` (score=0.7383) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotransferase deficiency T...
4. `medquad_ans_0000445-1__dup1` (score=0.7290) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of arginine in the blo...
5. `medquad_ans_0000065-3` (score=0.7280) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This cycle processes exce...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families. [medquad_ans_0000066-2]

**Citations detected**: `medquad_ans_0000066-2`
**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

## Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Top docs**:

1. `medquad_ans_0000379-2` (score=0.7192) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0000379-5` (score=0.6771) - These resources address the diagnosis or management of focal dermal hypoplasia: - Gene Review: Gene Review: Focal Dermal Hypoplasia - Genetic Testing Registry: Focal dermal hypoplasia - MedlinePlus Encyclopedia: Ectodermal dysplasia - Me...
3. `medquad_ans_0002399-3` (score=0.6637) - How is this condition inherited? Focal dermal hypoplasia is caused by mutations in the PORCN gene and is inherited in an X-linked dominant manner. Many cases of focal dermal hypoplasia result from a new mutation and occur in people with ...
4. `medquad_ans_0000379-1` (score=0.6633) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. Although intelligenc...
5. `medquad_ans_0002399-2` (score=0.6551) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, although almost all affect...

**Answer**:

The symptoms of focal dermal hypoplasia include skin abnormalities present from birth such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), small clusters of veins on the surface of the skin (telangiectases), and streaks of slightly darker or lighter skin. These skin changes may cause pain, itching, irritation, or lead to skin infections. Wart-like growths called papillomas are usually not present at birth but develop with age [medquad_ans_0002399-3].

**Citations detected**: `medquad_ans_0002399-3`
**Supported citations**: `medquad_ans_0002399-3`
**Unsupported citations**: _none_

## Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.6607) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
2. `medquad_ans_0000418-5__dup1` (score=0.6575) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild game meat. Laborato...
3. `medquad_ans_0000418-6` (score=0.6520) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision to treat based upon...
4. `medquad_ans_0000432-1__dup3` (score=0.6463) - Whipworm (Trichuris trichiura) is an intestinal parasite of humans. The larvae and adult worms live in the intestine of humans and can cause intestinal disease. The name is derived from the worm’s distinctive whip-like shape.
5. `medquad_ans_0000265-4__dup2` (score=0.6431) - Loiasis is an infection caused by the parasitic worm Loa loa.

**Answer**:

People at risk for trichinellosis include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking puts individuals at risk. Additionally, individuals who consume raw or undercooked wild game, especially bear meat, are at risk. The risk from commercially raised and properly prepared pork is very low, but it remains a concern with raw or undercooked wild game [medquad_ans_0000418-2__dup1].

**Citations detected**: `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Top docs**:

1. `medquad_ans_0000108-4__dup1` (score=0.6591) - The NINDS conducts and supports a broad program of basic and clinical research on all types of epilepsy, including Dravet syndrome. Study of the genetic defects responsible for Dravet syndrome and related disorders is expected to lead to...
2. `medquad_ans_0000108-3__dup1` (score=0.6471) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of cognitive impairment a...
3. `medquad_ans_0001924-1` (score=0.6420) - Dravet syndrome is a severe form of epilepsy. The condition appears during the first year of life as frequent fever-related (febrile) seizures. As the condition progresses, other types of seizures typically occur, including myoclonus and...
4. `medquad_ans_0000108-2__dup2` (score=0.6341) - Seizures in Dravet syndrome are difficult to control, but can be reduced by anticonvulsant drugs. A ketogenic diet, high in fats and low in carbohydrates, also may be beneficial.
5. `medquad_ans_0000548-2` (score=0.6313) - Juvenile myoclonic epilepsy affects an estimated 1 in 1,000 people worldwide. Approximately 5 percent of people with epilepsy have juvenile myoclonic epilepsy.

**Answer**:

Symptoms of Dravet syndrome include frequent fever-related (febrile) seizures that appear during the first year of life, progression to other types of seizures such as myoclonus and status epilepticus, deterioration of intellectual development around age 2, lack of coordination, poor language development, hyperactivity, and difficulty relating to others. The degree of cognitive impairment correlates with the frequency of seizures [medquad_ans_0001924-1]. Additionally, as children with Dravet syndrome get older, their decline in cognitive function stabilizes and may improve slightly, though most teenagers remain dependent on caregivers [medquad_ans_0000108-3]. In 30 to 80 percent of cases, the condition is caused by changes in the SCN1A gene, which affects brain cell function [medquad_ans_0001924-1]. Seizures are difficult

**Citations detected**: `medquad_ans_0001924-1`, `medquad_ans_0000108-3`
**Supported citations**: `medquad_ans_0001924-1`
**Unsupported citations**: `medquad_ans_0000108-3`

## Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Top docs**:

1. `medquad_ans_0000826-4` (score=0.6899) - Familial forms of prion disease are inherited in an autosomal dominant pattern, which means one copy of the altered PRNP gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the altered gene f...
2. `medquad_ans_0000826-1` (score=0.6797) - Prion disease represents a group of conditions that affect the nervous system in humans and animals. In people, these conditions impair brain function, causing changes in memory, personality, and behavior; a decline in intellectual funct...
3. `medquad_ans_0000826-2` (score=0.6775) - These disorders are very rare. Although the exact prevalence of prion disease is unknown, studies suggest that this group of conditions affects about one person per million worldwide each year. Approximately 350 new cases are reported an...
4. `medquad_ans_0002533-3` (score=0.6703) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although the exact function ...
5. `medquad_ans_0001663-1` (score=0.6701) - Creutzfeldt-Jakob disease (CJD) is a rare fatal brain disorder that usually occurs later in life and runs a rapid course. In the early stages of the disease, patients may have failing memory, behavior changes, impaired coordination, and ...

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which encode the prion protein. These mutations result in the production of an abnormally shaped prion protein that accumulates in the brain, forming clumps that damage or destroy neurons [medquad_ans_0002533-3]. Familial forms of prion disease are inherited in an autosomal dominant pattern, meaning one copy of the altered PRNP gene is sufficient to cause the disorder, and such mutations can occur either inherited from an affected parent or as a new mutation during reproductive cell formation or early embryonic development [medquad_ans_0000826-4].

**Citations detected**: `medquad_ans_0002533-3`, `medquad_ans_0000826-4`
**Supported citations**: `medquad_ans_0002533-3`, `medquad_ans_0000826-4`
**Unsupported citations**: _none_

## Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.6759) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information is available, the ta...
2. `medquad_ans_0000304-2` (score=0.6423) - Duane-radial ray syndrome is a rare condition whose prevalence is unknown. Only a few affected families have been reported worldwide.
3. `medquad_ans_0000156-2` (score=0.6383) - Carpenter syndrome is thought to be a rare condition; approximately 70 cases have been described in the scientific literature.
4. `medquad_ans_0001553-1` (score=0.6380) - Congenital radio-ulnar synostosis is a rare condition in which there is an abnormal connection (synostosis) of the radius and ulna (bones in the forearm) at birth. The condition is present in both arms (bilateral) in approximately 60% of...
5. `medquad_ans_0000256-2` (score=0.6260) - Crouzonodermoskeletal syndrome is rare; this condition is seen in about 1 person per million.

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, hypoplasia of the radius, oligodactyly (hands), proximal placement of thumb, proximal radial head dislocation, cleft palate, cleft upper lip, high palate, hydronephrosis, hypotelorism, long face, narrow face, and mesomelia [medquad_ans_0006430-1].

**Citations detected**: `medquad_ans_0006430-1`
**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

## Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.6708) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the table below includes h...
2. `medquad_ans_0001033-2` (score=0.6366) - Werner syndrome is estimated to affect 1 in 200,000 individuals in the United States. This syndrome occurs more often in Japan, affecting 1 in 20,000 to 1 in 40,000 people.
3. `medquad_ans_0000033-3__dup1` (score=0.6240) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some individuals with lat...
4. `medquad_ans_0000269-1__dup1` (score=0.6109) - Alzheimer disease (AD) is a degenerative disease of the brain that causes gradual loss of memory, judgment, and the ability to function socially. Alzheimer disease currently affects about 5 million people. About 75 percent of Alzheimer d...
5. `medquad_ans_0000017-3` (score=0.6105) - In very few families, people develop AD in their 30s, 40s, and 50s. This is known as "early onset" AD. These individuals have a mutation in one of three different inherited genes that causes the disease to begin at an earlier age. More t...

**Answer**:

The signs and symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology (HPO), which compiles information from sources such as Orphanet, with frequencies often listed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Citations detected**: `medquad_ans_0005029-1`
**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

## Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Top docs**:

1. `medquad_ans_0000979-1` (score=0.7301) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (platelets). This plate...
2. `medquad_ans_0000979-5` (score=0.7192) - These resources address the diagnosis or management of TAR syndrome: - Gene Review: Gene Review: Thrombocytopenia Absent Radius Syndrome - Genetic Testing Registry: Radial aplasia-thrombocytopenia syndrome - MedlinePlus Encyclopedia: Ske...
3. `medquad_ans_0000881-2` (score=0.7160) - Roberts syndrome is a rare disorder; approximately 150 affected individuals have been reported.
4. `medquad_ans_0006007-1` (score=0.7042) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to normal over time. In...
5. `medquad_ans_0000304-2` (score=0.7036) - Duane-radial ray syndrome is a rare condition whose prevalence is unknown. Only a few affected families have been reported worldwide.

**Answer**:

The provided context does not specify the exact number of people affected by thrombocytopenia-absent radius (TAR) syndrome. While it mentions that TAR syndrome is unusual and associated with specific features, no population statistics or prevalence data are given.

Answer: Insufficient information to determine the number of people affected by thrombocytopenia-absent radius syndrome. [medquad_ans_0000979-1]

**Citations detected**: `medquad_ans_0000979-1`
**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

## Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Top docs**:

1. `medquad_ans_0000972-5` (score=0.7220) - These resources address the diagnosis or management of Tay-Sachs disease: - Gene Review: Gene Review: Hexosaminidase A Deficiency - Genetic Testing Registry: Tay-Sachs disease - MedlinePlus Encyclopedia: Tay-Sachs Disease These resources...
2. `medquad_ans_0000250-4__dup1` (score=0.7103) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological disease. The NINDS is a...
3. `medquad_ans_0000265-3__dup1` (score=0.7061) - Even with the best of care, children with Tay-Sachs disease usually die by age 4, from recurring infection.
4. `medquad_ans_0000265-4__dup1` (score=0.7037) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological disease. The NINDS is a...
5. `medquad_ans_0000430-5` (score=0.7015) - These resources address the diagnosis or management of GM2-gangliosidosis, AB variant: - Genetic Testing Registry: Tay-Sachs disease, variant AB These resources from MedlinePlus offer information about the diagnosis and management of var...

**Answer**:

Research on Tay-Sachs disease includes the development of biomarkers using variations of magnetic resonance imaging to evaluate brain biochemistry and disease progression. Additionally, NINDS-funded research is exploring virus-delivered gene therapy, initially tested in animal models of Tay-Sachs and Sandhoff diseases, with the goal of translating this approach for use in humans [medquad_ans_0000250-4__dup1]. This research is part of the broader efforts by the Lysosomal Disease Network to improve diagnosis, management, and therapy of rare diseases like Tay-Sachs [medquad_ans_0000250-4__dup1].

**Citations detected**: `medquad_ans_0000250-4__dup1`
**Supported citations**: `medquad_ans_0000250-4__dup1`
**Unsupported citations**: _none_

## Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Top docs**:

1. `medquad_ans_0000008-2` (score=0.6730) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on one of the Eastern Ca...
2. `medquad_ans_0000212-1` (score=0.6514) - Cone-rod dystrophy is a group of related eye disorders that causes vision loss, which becomes more severe over time. These disorders affect the retina, which is the layer of light-sensitive tissue at the back of the eye. In people with c...
3. `medquad_ans_0000008-1` (score=0.6513) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder...
4. `medquad_ans_0000204-2` (score=0.6456) - Red-green color vision defects are the most common form of color vision deficiency. This condition affects males much more often than females. Among populations with Northern European ancestry, it occurs in about 1 in 12 males and 1 in 2...
5. `medquad_ans_0001049-1` (score=0.6420) - X-linked congenital stationary night blindness is a disorder of the retina, which is the specialized tissue at the back of the eye that detects light and color. People with this condition typically have difficulty seeing in low light (ni...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals can also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life [medquad_ans_0000008-1]. Complete achromatopsia is more common than incomplete achromatopsia and occurs frequently among Pingelapese islanders, where between

**Citations detected**: `medquad_ans_0000008-1`
**Supported citations**: `medquad_ans_0000008-1`
**Unsupported citations**: _none_

## Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Top docs**:

1. `medquad_ans_0000039-9` (score=0.6915) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
2. `medquad_ans_0000544-1__dup1` (score=0.6788) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments are not helping you a...
3. `medquad_ans_0000039-13` (score=0.6743) - Knee replacement may be either total or partial/unicompartmental. In total knee replacement, as the name suggests, the entire knee joint is replaced. You will likely need a total knee replacement if you have damage to several parts of yo...
4. `medquad_ans_0000039-2__dup2` (score=0.6480) - There are many different types and designs of artificial knees. Most consist of three components: - the femoral component, which is the part that attaches to the thigh bone - the tibial component, the part that attaches to the shin bone ...
5. `medquad_ans_0000051-16` (score=0.6324) - Yes. Some complications from Paget's disease respond well to surgery. Joint replacement may be helpful in people with severe arthritis of the hip or knee. Surgery can also realign affected leg bones to reduce pain or help broken bones he...

**Answer**:

The context provided does not mention treatments for knee replacement; it describes the reasons for undergoing knee replacement, the types of surgery (total or partial), the components of artificial knees, and potential risks. Therefore, based on the given information, there is no answer to the question about treatments for knee replacement.

Answer: Insufficient context to determine treatments for knee replacement.

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Top docs**:

1. `medquad_ans_0001004-2` (score=0.6739) - Type A insulin resistance syndrome is estimated to affect about 1 in 100,000 people worldwide. Because females have more health problems associated with the condition, it is diagnosed more often in females than in males.
2. `medquad_ans_0000044-2__dup2` (score=0.6681) - Insulin resistance is a condition in which the body produces insulin but does not use it effectively. When people have insulin resistance, glucose builds up in the blood instead of being absorbed by the cells, leading to type 2 diabetes ...
3. `medquad_ans_0000090-3__dup2` (score=0.6551) - Metabolic syndrome has several causes that act together. You can control some of the causes, such as overweight and obesity, an inactive lifestyle, and insulin resistance. You can't control other factors that may play a role in causing m...
4. `medquad_ans_0000608-1__dup1` (score=0.6513) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, a type of fat, in yo...
5. `medquad_ans_0000032-6__dup1` (score=0.6475) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of the following five ...

**Answer**:

The provided context does not describe the symptoms of metabolic syndrome. It lists the conditions associated with metabolic syndrome, such as high blood pressure, high blood glucose, high triglyceride levels, low HDL cholesterol, and excess fat around the waist, but does not explicitly state these as symptoms. Therefore, based on the given information, the symptoms of metabolic syndrome cannot be accurately identified.

[medquad_ans_0000608-1__dup1] [medquad_ans_0000090-3__dup2]

**Citations detected**: `medquad_ans_0000608-1__dup1`, `medquad_ans_0000090-3__dup2`
**Supported citations**: `medquad_ans_0000608-1__dup1`, `medquad_ans_0000090-3__dup2`
**Unsupported citations**: _none_

## Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Top docs**:

1. `medquad_ans_0000501-2` (score=0.6489) - Hypochondrogenesis and achondrogenesis, type 2 (a similar skeletal disorder) together affect 1 in 40,000 to 60,000 newborns.
2. `medquad_ans_0000141-1__dup4` (score=0.6364) - An ectopic kidney is a birth defect in which a kidney is located below, above, or on the opposite side of its usual position. About one in 900 people has an ectopic kidney.1
3. `medquad_ans_0000867-2` (score=0.6353) - Renal tubular dysgenesis is a rare disorder, but its prevalence is unknown.
4. `medquad_ans_0000501-4` (score=0.6281) - Hypochondrogenesis is considered an autosomal dominant disorder because one copy of the altered gene in each cell is sufficient to cause the condition. It is caused by new mutations in the COL2A1 gene and occurs in people with no history...
5. `medquad_ans_0000141-9` (score=0.6260) - - An ectopic kidney is a birth defect in which a kidney is located below, above, or on the opposite side of its usual position. - Factors that may lead to an ectopic kidney include - poor development of a kidney bud - a defect in the kid...

**Answer**:

The provided context does not contain any information about the symptoms of Pyknoachondrogenesis. Therefore, it is not possible to answer the question based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Top docs**:

1. `medquad_ans_0003965-3` (score=0.6892) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for an individual to be ...
2. `medquad_ans_0000647-5` (score=0.6823) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Registry: Sphingolipid...
3. `medquad_ans_0000181-3__dup1` (score=0.6684) - The prognosis for the leukodystrophies varies according to the specific type of leukodystrophy.
4. `medquad_ans_0000647-2` (score=0.6616) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigrated to Israel from s...
5. `medquad_ans_0000192-1__dup1` (score=0.6591) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage materials in cells i...

**Answer**:

The provided context does not include information about the symptoms of metachromatic leukodystrophy. [medquad_ans_0003965-3], [medquad_ans_0000647-5], [medquad_ans_0000181-3__dup1], [medquad_ans_0000647-2], and [medquad_ans_0000192-1__dup1] discuss inheritance, prevalence, diagnosis, management, and general characteristics of the condition but do not specify symptoms. Therefore, based on the given context, the symptoms of metachromatic leukodystrophy cannot be determined.

**Citations detected**: `medquad_ans_0003965-3`, `medquad_ans_0000647-5`, `medquad_ans_0000181-3__dup1`, `medquad_ans_0000647-2`, `medquad_ans_0000192-1__dup1`
**Supported citations**: `medquad_ans_0003965-3`, `medquad_ans_0000647-5`, `medquad_ans_0000181-3__dup1`, `medquad_ans_0000647-2`, `medquad_ans_0000192-1__dup1`
**Unsupported citations**: _none_

## Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Top docs**:

1. `medquad_ans_0000186-5` (score=0.6906) - These resources address the diagnosis or management of chronic granulomatous disease: - American Academy of Allergy, Asthma, and Immunology - Gene Review: Gene Review: Chronic Granulomatous Disease - Genetic Testing Registry: Chronic gra...
2. `medquad_ans_0000186-2` (score=0.6745) - Chronic granulomatous disease is estimated to occur in 1 in 200,000 to 250,000 people worldwide.
3. `medquad_ans_0000186-4` (score=0.6554) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who...
4. `medquad_ans_0000242-4` (score=0.6395) - This condition is reported to have an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of ...
5. `medquad_ans_0001001-4` (score=0.6366) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, some people who inherit the altered gene never develop features of TRAPS. (...

**Answer**:

Yes, chronic granulomatous disease is inherited. It can be inherited in an X-linked recessive pattern when caused by mutations in the CYBB gene, or in an autosomal recessive pattern when caused by mutations in the CYBA, NCF1, NCF2, or NCF4 genes [medquad_ans_0000186-4].

**Citations detected**: `medquad_ans_0000186-4`
**Supported citations**: `medquad_ans_0000186-4`
**Unsupported citations**: _none_

## Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000061-4__dup1` (score=0.7450) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research focuses on finding ...
2. `medquad_ans_0000222-4__dup1` (score=0.6917) - The NINDS supports research on disorders of the brain, spinal cord, and peripheral nerves that can cause paresthesia. The goals of this research are to increase scientific understanding of these disorders and to find ways to prevent, tre...
3. `medquad_ans_0000255-4__dup1` (score=0.6905) - The NINDS supports research on brain and spinal cord disorders that can cause spasticity. The goals of this research are to increase scientific understanding about these disorders and to find ways to prevent, treat, and cure them.
4. `medquad_ans_0000062-4__dup1` (score=0.6888) - The NINDS vigorously pursues a research program seeking new treatments for chronic pain and nervous system damage. The goals of this research are to develop ways to more effectively treat and potentially reverse debilitating conditions s...
5. `medquad_ans_0000266-4__dup1` (score=0.6888) - The NINDS conducts and supports research on disorders of the spinal cord. The goals of this research are to find ways to prevent, treat, and cure these disorders.

**Answer**:

The provided context does not contain information about the outlook for Central Cord Syndrome. Therefore, it is not possible to determine the outlook based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Top docs**:

1. `medquad_ans_0000094-4__dup3` (score=0.7431) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting syndrome, which occu...
2. `medquad_ans_0000261-2` (score=0.7271) - The exact prevalence of cyclic vomiting syndrome is unknown; estimates range from 4 to 2,000 per 100,000 children. The condition is diagnosed less frequently in adults, although recent studies suggest that the condition may begin in adul...
3. `medquad_ans_0000094-1__dup6` (score=0.7126) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. The ep...
4. `medquad_ans_0000094-13` (score=0.6840) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. - Th...
5. `medquad_ans_0000261-3` (score=0.6780) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnormalities, and gastro...

**Answer**:

The cause of cyclic vomiting syndrome is unknown [medquad_ans_0000094-13].

**Citations detected**: `medquad_ans_0000094-13`
**Supported citations**: `medquad_ans_0000094-13`
**Unsupported citations**: _none_

## Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Top docs**:

1. `medquad_ans_0000691-1__dup1` (score=0.7218) - If you're a parent, you get plenty of suggestions on how to raise your child. From experts to other parents, people are always ready to offer advice. Parenting tips, parents' survival guides, dos, don'ts, shoulds and shouldn'ts - new one...
2. `medquad_ans_0000603-1__dup1` (score=0.7176) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and make choices. Mental he...
3. `medquad_ans_0000071-1__dup3` (score=0.7166) - Is it hard for your child to sit still? Does your child act without thinking first? Does your child start but not finish things? If so, your child may have attention deficit hyperactivity disorder (ADHD). Nearly everyone shows some of th...
4. `medquad_ans_0000180-1__dup2` (score=0.6984) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not always easy to know when...
5. `medquad_ans_0000189-1__dup2` (score=0.6896) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise and insuring their ...

**Answer**:

Yes, there is information about child mental health. Mental health includes emotional, psychological, and social well-being and affects how children think, feel, and act as they cope with life. It is important at every stage of life, including childhood and adolescence. Mental illnesses can affect a child's thinking, mood, and behavior and may have causes related to genetics, family history, life experiences such as stress or abuse, and biological factors. Early recognition and treatment of mental illnesses in children are important because once they develop, they become a regular part of the child's behavior and are more difficult to treat [medquad_ans_0000603-1__dup1]. Additionally, everyday stresses can cause changes in a child's behavior, making it sometimes hard to determine when a serious problem exists [medquad_ans_0000180-1__dup2].

**Citations detected**: `medquad_ans_0000603-1__dup1`, `medquad_ans_0000180-1__dup2`
**Supported citations**: `medquad_ans_0000603-1__dup1`, `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

## Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Top docs**:

1. `medquad_ans_0001618-6` (score=0.7068) - How might Cowden syndrome be treated? Because Cowden syndrome is associated with an increased risk for certain types of cancer, management is typically focused on high-risk cancer screening. According to the National Comprehensive Cancer...
2. `medquad_ans_0000246-2` (score=0.6959) - Although the exact prevalence of Cowden syndrome is unknown, researchers estimate that it affects about 1 in 200,000 people.
3. `medquad_ans_0000246-4` (score=0.6855) - Cowden syndrome and Cowden-like syndrome are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the condition and increase the risk of developing cancer. In some cases...
4. `medquad_ans_0004195-5` (score=0.6718) - How is multiple familial trichoepithelioma diagnosed? Diagnosis of multiple familial trichoepithelioma is made based upon the clinical symptoms in the patient, the patients family history, and the appearance of the trichoepithelioma cell...
5. `medquad_ans_0000246-1` (score=0.6703) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas. These growths are m...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous, tumor-like growths called hamartomas. These growths are most commonly found on the skin and mucous membranes (such as the lining of the mouth and nose), but they can also occur in the intestine and other parts of the body [medquad_ans_0000246-1].

**Citations detected**: `medquad_ans_0000246-1`
**Supported citations**: `medquad_ans_0000246-1`
**Unsupported citations**: _none_

## Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0000395-2` (score=0.7087) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dystrophy has an estima...
2. `medquad_ans_0000395-5` (score=0.6854) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscular Dystrophy - Geneti...
3. `medquad_ans_0000306-2` (score=0.6535) - Duchenne and Becker muscular dystrophies together affect 1 in 3,500 to 5,000 newborn males worldwide. Between 400 and 600 boys in the United States are born with these conditions each year.
4. `medquad_ans_0000666-5` (score=0.6526) - These resources address the diagnosis or management of Miyoshi myopathy: - Gene Review: Gene Review: ANO5-Related Muscle Diseases - Gene Review: Gene Review: Dysferlinopathy - Genetic Testing Registry: Miyoshi muscular dystrophy 1 - Gene...
5. `medquad_ans_0000205-4__dup1` (score=0.6483) - The NINDS supports a broad program of research studies on MD. The goals of these studies are to understand MD and to develop techniques to diagnose, treat, prevent, and ultimately cure the disorder. The NINDS is a member of the Muscular ...

**Answer**:

The provided context does not specify the treatments for Fukuyama congenital muscular dystrophy. While it mentions resources that address the diagnosis or management of the condition, such as diagnostic tests, genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and genetic testing registry entries, it does not detail specific treatments [medquad_ans_0000395-5]. Therefore, based on the given information, the exact treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Citations detected**: `medquad_ans_0000395-5`
**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

## Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Top docs**:

1. `medquad_ans_0000004_7-7` (score=0.6512) - Key Points - There are different types of treatment for children with non-Hodgkin lymphoma. - Children with non-Hodgkin lymphoma should have their treatment planned by a team of doctors who are experts in treating childhood cancer. - Som...
2. `medquad_ans_0000004_6-8` (score=0.6481) - Key Points - There are different types of treatment for children with Hodgkin lymphoma. - Children with Hodgkin lymphoma should have their treatment planned by a team of health care providers who are experts in treating childhood cancer....
3. `medquad_ans_0000004_7-1` (score=0.6399) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major types of childhood n...
4. `medquad_ans_0000004_6-5` (score=0.6379) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whether there are B symp...
5. `medquad_ans_0000004_6-6` (score=0.6339) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spreads in the body. - Sta...

**Answer**:

The provided context does not contain information on how to diagnose Childhood Hodgkin Lymphoma. It discusses treatment options for children with Non-Hodgkin Lymphoma, not diagnosis of Hodgkin Lymphoma. [medquad_ans_0000004_7-7]

**Citations detected**: `medquad_ans_0000004_7-7`
**Supported citations**: `medquad_ans_0000004_7-7`
**Unsupported citations**: _none_

## Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000556-2` (score=0.7226) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symptoms can be mild and...
2. `medquad_ans_0000556-5` (score=0.6945) - These resources address the diagnosis or management of KBG syndrome: - Genetic Testing Registry: KBG syndrome These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnosti...
3. `medquad_ans_0000556-3` (score=0.6817) - KBG syndrome is caused by mutations in the ANKRD11 gene. The protein produced from this gene enables other proteins to interact with each other and helps control gene activity. The ANKRD11 protein is found in nerve cells (neurons) in the...
4. `medquad_ans_0003361-1` (score=0.6718) - KBG syndrome is a rare condition characterized mainly by skeletal abnormalities, distinctive facial features, and intellectual disability. Specific signs and symptoms may include delayed bone age; abnormalities of the bones of the spine,...
5. `medquad_ans_0000556-1` (score=0.6659) - KBG syndrome is a rare disorder that affects several body systems. "KBG" represents the surname initials of the first families diagnosed with the disorder. Common signs and symptoms in individuals with this condition include unusual faci...

**Answer**:

Yes, KBG syndrome can be inherited. It is sometimes inherited in an autosomal dominant manner, particularly when caused by a mutation in the ANKRD11 gene. In some cases, individuals inherit the condition from a parent, while in other cases, it occurs sporadically without a family history [medquad_ans_0003361-1].

**Citations detected**: `medquad_ans_0003361-1`
**Supported citations**: `medquad_ans_0003361-1`
**Unsupported citations**: _none_

## Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Top docs**:

1. `medquad_ans_0000029-4` (score=0.7324) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an au...
2. `medquad_ans_0000029-2` (score=0.7221) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
3. `medquad_ans_0000028-2` (score=0.6830) - Aicardi syndrome is a very rare disorder. It occurs in about 1 in 105,000 to 167,000 newborns in the United States. Researchers estimate that there are approximately 4,000 affected individuals worldwide.
4. `medquad_ans_0000029-5` (score=0.6745) - These resources address the diagnosis or management of Aicardi-Goutieres syndrome: - Gene Review: Gene Review: Aicardi-Goutieres Syndrome - Genetic Testing Registry: Aicardi Goutieres syndrome - Genetic Testing Registry: Aicardi Goutiere...
5. `medquad_ans_0000029-1` (score=0.6619) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. However, about 20 percent ...

**Answer**:

Aicardi-Goutieres syndrome is a rare disorder that mainly affects the brain, the immune system, and the skin. Most newborns do not show signs or symptoms at birth, although about 20 percent are born with a combination of features including an enlarged liver and spleen (hepatosplenomegaly), elevated blood levels of liver enzymes, decreased blood platelets (thrombocytopenia), and abnormal neurological responses. This combination is typically associated with the immune system's response to congenital viral infection, even though no actual infection is found, leading to the syndrome being referred to as a "mimic of congenital infection." Within the first year of life, most individuals experience an episode of severe brain dysfunction (encephalopathy), usually lasting several months [medquad_ans_0000029-1]. The disorder can be inherited in an autosomal recessive pattern, where both copies of

**Citations detected**: `medquad_ans_0000029-1`
**Supported citations**: `medquad_ans_0000029-1`
**Unsupported citations**: _none_

## Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Top docs**:

1. `medquad_ans_0000026_1-7` (score=0.7227) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy - New types of treat...
2. `medquad_ans_0000026_3-7` (score=0.6855) - Key Points - There are different types of treatment for patients with Wilms tumor and other childhood kidney tumors. - Children with Wilms tumor or other childhood kidney tumors should have their treatment planned by a team of health car...
3. `medquad_ans_0000026_2-7` (score=0.6808) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tested in clinical tria...
4. `medquad_ans_0000028_3-8` (score=0.6758) - New types of treatment are being tested in clinical trials. Patients may want to think about taking part in a clinical trial. For some patients, taking part in a clinical trial may be the best treatment choice. Clinical trials are part o...
5. `medquad_ans_0000040_1-6` (score=0.6747) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about clinical trials is a...

**Answer**:

Research involving clinical trials is being conducted for renal cell cancer to improve current treatments or obtain information on new treatments. These clinical trials are designed to test new therapies and may be open to patients before, during, or after starting their cancer treatment. When clinical trials demonstrate that a new treatment is more effective than the standard treatment, it may become the new standard of care [medquad_ans_0000026_1-7].

**Citations detected**: `medquad_ans_0000026_1-7`
**Supported citations**: `medquad_ans_0000026_1-7`
**Unsupported citations**: _none_

## Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Top docs**:

1. `medquad_ans_0000824-2` (score=0.6703) - An estimated 1 in 10,000 people have primary sclerosing cholangitis, and the condition is diagnosed in approximately 1 in 100,000 people per year worldwide.
2. `medquad_ans_0000191-8` (score=0.6666) - - Primary sclerosing cholangitis (PSC) inflames, scars, and blocks bile ducts inside and outside the liver. - When bile ducts become blocked, bile builds up in the liver and damages liver cells. - PSC can lead to vitamin deficiencies, in...
3. `medquad_ans_0000190-6` (score=0.6586) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the progression of live...
4. `medquad_ans_0000257-2` (score=0.6569) - Cirrhosis affects more than 600,000 people in the United States; cryptogenic cirrhosis likely accounts for 5 to 30 percent of these cases.
5. `medquad_ans_0000824-1` (score=0.6528) - Primary sclerosing cholangitis is a condition that affects the bile ducts. These ducts carry bile (a fluid that helps to digest fats) from the liver, where bile is produced, to the gallbladder, where it is stored, and to the small intest...

**Answer**:

Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages, treatment can slow the progression of liver damage to cirrhosis. In the early stages of cirrhosis, the goals are to slow tissue scarring and prevent complications. As cirrhosis progresses, additional treatments and hospitalization may be needed to manage complications.

Medications: Health care providers prescribe ursodiol (Actigall, Urso) to treat primary biliary cirrhosis. Ursodiol is a nontoxic bile acid taken orally that replaces the more toxic bile acids normally produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival. Although early treatment provides the most benefit, ursodiol can still slow liver damage even when used late in the

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.6882) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People who have seizure dis...
2. `medquad_ans_0000023-1__dup1` (score=0.6826) - Cerebral hypoxia refers to a condition in which there is a decrease of oxygen supply to the brain even though there is adequate blood flow. Drowning, strangling, choking, suffocation, cardiac arrest, head trauma, carbon monoxide poisonin...
3. `medquad_ans_0000967-1__dup1` (score=0.6574) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoid alcohol when swimm...
4. `medquad_ans_0000916-1__dup2` (score=0.6414) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of Earth. A tsunami ca...
5. `medquad_ans_0000977-1__dup1` (score=0.6361) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from minor to life-threate...

**Answer**:

Drowning is a condition that occurs when a person takes in too much water, leading to a lack of oxygen in the lungs. It can happen quickly and silently, even with as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most at risk in swimming pools. People with seizure disorders are also at increased risk in the water. Drowning can result in cerebral hypoxia, a condition where the brain does not receive enough oxygen, which can lead to brain cell death within five minutes of oxygen deprivation. In severe cases, prolonged hypoxia can cause coma, seizures, or even brain death [medquad_ans_0000299-1\_\_dup2][medquad_ans_0000023-1__dup1]. Drowning is also the most common cause of death related to tsunamis [medquad_ans_00009

**Citations detected**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000023-1__dup1`
**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000023-1__dup1`
**Unsupported citations**: _none_

## Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Top docs**:

1. `medquad_ans_0002058-4` (score=0.6923) - Is Ehlers-Danlos Syndrome, kyphoscoliotic type inherited? Ehlers-Danlos syndrome, kyphoscoliosis type is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the resp...
2. `medquad_ans_0002049-6` (score=0.6877) - How might Ehlers-Danlos syndrome be treated? There is no specific cure for Ehlers-Danlos syndrome (EDS). The treatment and management is focused on preventing serious complications and relieving associated signs and symptoms. Because the...
3. `medquad_ans_0002058-1` (score=0.6794) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragile and bruises easil...
4. `medquad_ans_0002058-5` (score=0.6692) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tests may then be recom...
5. `medquad_ans_0000313-2` (score=0.6686) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classical forms are most co...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen, specifically due to mutations in the PLOD1 gene. It is inherited in an autosomal recessive manner, meaning a person must have mutations in both copies of the PLOD1 gene to be affected. Carriers, who have one mutated copy, typically do not show signs or symptoms. Common signs and symptoms include hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera [medquad_ans_0002058-1]. Diagnosis is typically based on characteristic signs and symptoms, and may be confirmed with urine tests, skin biopsy to detect deficiencies in collagen-forming enzymes, or genetic testing for mutations in the P

**Citations detected**: `medquad_ans_0002058-1`
**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

## Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Top docs**:

1. `medquad_ans_0000898-2` (score=0.6662) - Sepiapterin reductase deficiency appears to be a rare condition. At least 30 cases have been described in the scientific literature.
2. `medquad_ans_0000898-5` (score=0.6644) - These resources address the diagnosis or management of sepiapterin reductase deficiency: - Gene Review: Gene Review: Sepiapterin Reductase Deficiency - Genetic Testing Registry: Sepiapterin reductase deficiency These resources from Medli...
3. `medquad_ans_0001862-6` (score=0.6582) - How might dihydropyrimidine dehydrogenase deficiency be treated in infants and children? Currently, no treatment or cure exists for the inborn error of metabolism form of DHD deficiency. Symptoms usually remain the same throughout the pe...
4. `medquad_ans_0000288-5` (score=0.6575) - These resources address the diagnosis or management of dihydropyrimidine dehydrogenase deficiency: - Genetic Testing Registry: Dihydropyrimidine dehydrogenase deficiency These resources from MedlinePlus offer information about the diagno...
5. `medquad_ans_0000286-5` (score=0.6526) - These resources address the diagnosis or management of dihydrolipoamide dehydrogenase deficiency: - Gene Review: Gene Review: Dihydrolipoamide Dehydrogenase Deficiency - Genetic Testing Registry: Maple syrup urine disease, type 3 These r...

**Answer**:

The provided context does not contain information about the symptoms of dihydropteridine reductase deficiency. Therefore, the symptoms cannot be determined from the given text. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.7384) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000669-4` (score=0.7120) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suf...
3. `medquad_ans_0000669-1` (score=0.6845) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin word for necklace (m...
4. `medquad_ans_0000669-3` (score=0.6687) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for making proteins that gi...
5. `medquad_ans_0000685-2` (score=0.6461) - MONA is rare; its prevalence is unknown. This condition has been reported in multiple populations worldwide.

**Answer**:

The prevalence of monilethrix is unknown. [medquad_ans_0000669-2]

**Citations detected**: `medquad_ans_0000669-2`
**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

## Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.7142) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the results of newborn s...
2. `medquad_ans_0002890-1` (score=0.6946) - Hirschsprung disease is a disease of the large intestine or colon. People with this disease do not have the nerve cells in the intestine required to expel stools from the body normally. Symptoms of Hirschsprung disease usually show up in...
3. `medquad_ans_0000481-2` (score=0.6857) - Hirschsprung disease occurs in approximately 1 in 5,000 newborns.
4. `medquad_ans_0002733-2` (score=0.6817) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headaches,and hypotonia Psy...
5. `medquad_ans_0002890-3` (score=0.6765) - What causes Hirschsprung disease? There are a number of different causes of Hirschsprung disease (HSCR). For example, HSCR may occur as: A part of a syndrome In association with a chromosome anomaly (such as trisomy 21 or Down syndrome) ...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria and is caused by mutations in the SLC6A19 gene, inherited in an autosomal recessive manner. Most people with Hartnup disease are asymptomatic, but those who do show symptoms typically experience onset between the ages of 3 and 9, occasionally in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often resolve spontaneously after 1–4 weeks. Treatment may include a high-protein diet

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Top docs**:

1. `medquad_ans_0000424-2` (score=0.6827) - GSD IV is estimated to occur in 1 in 600,000 to 800,000 individuals worldwide. Type IV accounts for roughly 3 percent of all cases of glycogen storage disease.
2. `medquad_ans_0002620-4` (score=0.6687) - How is glycogen storage disease type 4 inherited? Glycogen storage disease type 4 is inherited in an autosomal recessive manner. This means that an individual must have 2 abnormal copies of the GBE1 gene to be affected (one abnormal copy...
3. `medquad_ans_0000428-5` (score=0.6675) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resources from MedlinePlus ...
4. `medquad_ans_0000427-5` (score=0.6649) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These resources from Medl...
5. `medquad_ans_0002620-1` (score=0.6623) - Glycogen storage disease type 4 (GSD 4) is part of a group of disorders which lead to abnormal accumulation of glycogen (a storage form of glucose) in various parts of the body. Symptoms of GSD 4 usually begin in infancy and typically in...

**Answer**:

The provided context does not contain information about the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis or management of glycogen storage disease type VII, including genetic testing, but does not specify the genetic changes associated with it. Therefore, based on the given context, the genetic changes related to glycogen storage disease type VII cannot be determined.

**Citations detected**: `medquad_ans_0000428-5`
**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

## Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Top docs**:

1. `medquad_ans_0000977-2` (score=0.7437) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
2. `medquad_ans_0000977-1` (score=0.7146) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low number of red blood cell...
3. `medquad_ans_0006061-1` (score=0.7103) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this condition between i...
4. `medquad_ans_0000977-5` (score=0.7049) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megaloblastic anemia, thi...
5. `medquad_ans_0000977-3` (score=0.6791) - Mutations in the SLC19A2 gene cause thiamine-responsive megaloblastic anemia syndrome. This gene provides instructions for making a protein called thiamine transporter 1, which transports thiamine into cells. Thiamine is found in many di...

**Answer**:

The treatments for thiamine-responsive megaloblastic anemia syndrome include high doses of vitamin B1 (thiamine), which can treat the anemia. In some cases, thiamine treatment can also reduce the amount of insulin a person needs for diabetes management [medquad_ans_0000977-1]. However, it remains unclear whether thiamine treatment can improve or prevent hearing loss [medquad_ans_0000977-1].

**Citations detected**: `medquad_ans_0000977-1`
**Supported citations**: `medquad_ans_0000977-1`
**Unsupported citations**: _none_

## Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000804-2` (score=0.6796) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Autosomal dominant pol...
2. `medquad_ans_0000135-4__dup3` (score=0.6725) - Researchers do not fully understand what causes cysts to grow in the kidneys of people with CKD. The fact that these cysts occur only in the kidneys and not in other parts of the body, as in PKD, indicates that the processes that lead to...
3. `medquad_ans_0000804-3` (score=0.6622) - Mutations in the PKD1, PKD2, and PKHD1 genes cause polycystic kidney disease. Mutations in either the PKD1 or PKD2 gene can cause autosomal dominant polycystic kidney disease; PKD1 gene mutations cause ADPKD type 1, and PKD2 gene mutatio...
4. `medquad_ans_0000172-1__dup3` (score=0.6616) - Simple kidney cysts are abnormal, fluid-filled sacs that form in the kidneys. Simple kidney cysts are different from the cysts that develop when a person has polycystic kidney disease (PKD), which is a genetic disorder. Simple kidney cys...
5. `medquad_ans_0000135-2__dup2` (score=0.6565) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems with blood vessels in ...

**Answer**:

The provided context does not include specific treatment recommendations or actions to take for polycystic kidney disease. While it describes the genetic basis, prevalence, and mechanisms of cyst formation, it does not address management, treatment, or patient care strategies. Therefore, based on the given information, a definitive answer cannot be provided. [medquad_ans_0000804-2][medquad_ans_0000804-3][medquad_ans_0000135-4__dup3]

**Citations detected**: `medquad_ans_0000804-2`, `medquad_ans_0000804-3`, `medquad_ans_0000135-4__dup3`
**Supported citations**: `medquad_ans_0000804-2`, `medquad_ans_0000804-3`, `medquad_ans_0000135-4__dup3`
**Unsupported citations**: _none_

## Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.7191) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research efforts are focused ...
2. `medquad_ans_0000253-4__dup1` (score=0.7104) - The goals of research on disorders such as Sjgren's syndrome focus on increasing knowledge and understanding of the disorder, improving diagnostic techniques, testing interventions, and finding ways to treat, prevent, and cure the disease.
3. `medquad_ans_0000875-2` (score=0.7103) - Rheumatoid arthritis affects about 1.3 million adults in the United States. Worldwide, it is estimated to occur in up to 1 percent of the population. The disease is two to three times more common in women than in men, which may be relate...
4. `medquad_ans_0000061-12` (score=0.6948) - There is no single, definitive test for rheumatoid arthritis. Common tests for rheumatoid arthritis include - The rheumatoid factor test. Rheumatoid factor is an antibody that is present eventually in the blood of most people with rheuma...
5. `medquad_ans_0000040-23` (score=0.6940) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current standard of care. Clini...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly why some people experience more severe symptoms than others. Scientists are investigating possible triggers, such as infectious agents like viruses and bacteria, to determine how they may contribute to the disease in individuals with a genetic predisposition. Researchers are also exploring the complex relationships between hormonal, nervous, and immune systems, including how changes in steroid hormones like estrogen and testosterone during a person's lifetime may influence the development, improvement, or flares of the disease. Additionally, studies are examining why rheumatoid arthritis often improves during pregnancy [medquad_ans_0000061-5__dup2].

Efforts are also underway to develop drugs that can reduce inflammation and slow or stop the disease with minimal side effects. To find new therapies, investigators are working to identify specific infectious agents and understand their mechanisms of action [medquad_ans_0000061-

**Citations detected**: `medquad_ans_0000061-5__dup2`
**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

## Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.7443) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the ...
2. `medquad_ans_0002561-3` (score=0.7262) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
3. `medquad_ans_0002561-1` (score=0.6952) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an abnormally-shaped hea...
4. `medquad_ans_0000409-2` (score=0.6549) - Gitelman syndrome affects an estimated 1 in 40,000 people worldwide.
5. `medquad_ans_0005571-3` (score=0.6518) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of the disease, and other...

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is surgery, typically performed between 9 to 12 months of age, to increase the space in the anterior part of the skull. This surgery is based on the person's specific signs and symptoms. If other sutures besides the coronal suture are involved, additional surgeries may be required [medquad_ans_0002561-4]. Treatment has included surgical correction of craniosynostosis, with no reported issues with development or normal intelligence [medquad_ans_0002561-1].

**Citations detected**: `medquad_ans_0002561-4`, `medquad_ans_0002561-1`
**Supported citations**: `medquad_ans_0002561-4`, `medquad_ans_0002561-1`
**Unsupported citations**: _none_

## Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Top docs**:

1. `medquad_ans_0000868-2` (score=0.6821) - Renpenning syndrome is a rare disorder; its prevalence is unknown. More than 60 affected individuals in at least 15 families have been identified.
2. `medquad_ans_0000868-5` (score=0.6614) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: Center for Genetic D...
3. `medquad_ans_0005571-1` (score=0.6507) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration and frequency of the...
4. `medquad_ans_0000091-4__dup1` (score=0.6479) - NINDS supports research on Cushing's syndrome aimed at finding new ways to diagnose, treat, and cure the disorder.
5. `medquad_ans_0000091-2__dup2` (score=0.6393) - Treatment of Cushing's syndrome depends on the cause of excess cortisol. If the cause is long-term use of a medication being used to treat another disorder, the physician may reduce the dosage until symptoms are under control. Surgery or...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. [medquad_ans_0000868-5] lists resources that address the diagnosis or management of Renpenning syndrome, such as genetic testing, genetic counseling, and other related services, but it does not specify any treatments. Therefore, based on the available information, the treatments for Renpenning syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000868-5`
**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

## Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.6933) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence of about 1 in 2,500 i...
2. `medquad_ans_0001104-1` (score=0.6843) - Charcot-Marie-Tooth disease type 2F (CMT2F) is a genetic disorder of the peripheral nerves. The subtypes of CMT type 2 (including type 2F) have similar features and are distinguished only by their disease-causing genes. Signs and symptom...
3. `medquad_ans_0000173-5` (score=0.6815) - These resources address the diagnosis or management of Charcot-Marie-Tooth disease: - Gene Review: Gene Review: Charcot-Marie-Tooth Hereditary Neuropathy Overview - Gene Review: Gene Review: Charcot-Marie-Tooth Neuropathy Type 1 - Gene R...
4. `medquad_ans_0001104-2` (score=0.6722) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic between the ages of 5...
5. `medquad_ans_0001104-4` (score=0.6710) - How is Charcot-Marie-Tooth disease type 2F inherited? Charcot-Marie-Tooth disease type 2F is inherited in an autosomal dominant manner. This means that only one mutated copy of the gene in each cell is sufficient to cause the condition. ...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001104-1] discusses symptoms of CMT2F, but there is no mention of CMT2G. Therefore, the symptoms of CMT2G cannot be determined from the given context.

**Citations detected**: `medquad_ans_0001104-1`
**Supported citations**: `medquad_ans_0001104-1`
**Unsupported citations**: _none_

## Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.6626) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure when standing or walki...
2. `medquad_ans_0001393-1` (score=0.6527) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
3. `medquad_ans_0000696-1__dup2` (score=0.6211) - Pelvic pain occurs mostly in the lower abdomen area. The pain might be steady, or it might come and go. If the pain is severe, it might get in the way of your daily activities. If you're a woman, you might feel a dull pain during your pe...
4. `medquad_ans_0001393-4` (score=0.6079) - What treatment is available for coccygodynia? Treatment for coccygodynia generally falls into conservative management or surgical intervention categories. The conservative approach typically includes hot sitz baths, NSAIDs, stool softene...
5. `medquad_ans_0000108-3__dup3` (score=0.5947) - About 75 percent of people will have hemorrhoids at some point in their lives.1 Hemorrhoids are most common among adults ages 45 to 65.2 Hemorrhoids are also common in pregnant women.

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Citations detected**: `medquad_ans_0001393-2`
**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

## Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Top docs**:

1. `medquad_ans_0000654-2` (score=0.6589) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
2. `medquad_ans_0000736-2` (score=0.6483) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
3. `medquad_ans_0000102-2` (score=0.6470) - The prevalence of all forms of microcephaly that are present from birth (primary microcephaly) ranges from 1 in 30,000 to 1 in 250,000 newborns worldwide. About 200 families with MCPH have been reported in the medical literature. This co...
4. `medquad_ans_0003270-1` (score=0.6446) - Agenesis of the corpus callosum (ACC) is a birth defect in which the structure that connects the two sides of the brain (the corpus callosum) is partially or completely absent. This birth defect can occur as an isolated condition or in c...
5. `medquad_ans_0004025-1` (score=0.6444) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgenesis and cleft lip-p...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, microcephaly, preaxial hand polydactyly, proptosis, and seizures. These symptoms are based on data from the Human Phenotype Ontology, which compiles information on the frequency of signs and symptoms in individuals with this condition [medquad_ans_0004025-1].

**Citations detected**: `medquad_ans_0004025-1`
**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

## Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Top docs**:

1. `medquad_ans_0000170-3__dup1` (score=0.7010) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
2. `medquad_ans_0000562-2` (score=0.6871) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
3. `medquad_ans_0000562-5` (score=0.6614) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing Registry: Klippel-F...
4. `medquad_ans_0003402-5` (score=0.6432) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal a...
5. `medquad_ans_0000561-2` (score=0.6354) - Klinefelter syndrome affects 1 in 500 to 1,000 newborn males. Most variants of Klinefelter syndrome are much rarer, occurring in 1 in 50,000 or fewer newborns. Researchers suspect that Klinefelter syndrome is underdiagnosed because the c...

**Answer**:

Klippel-Feil syndrome is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Due to the wide range of associated abnormalities involving many body systems, other initial exams are necessary to detect additional physical abnormalities or underlying conditions. These include: examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each individual [medquad_ans_0003402-5].

**Citations detected**: `medquad_ans_0003402-5`
**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

## Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0000222-2` (score=0.7356) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
2. `medquad_ans_0001497-1` (score=0.6961) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severity from a thinned are...
3. `medquad_ans_0000222-4` (score=0.6659) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...
4. `medquad_ans_0000222-5` (score=0.6635) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia University Medical Center...
5. `medquad_ans_0001497-3` (score=0.6567) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic set of findings. Curre...

**Answer**:

Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severity from a thinned area in the diaphragm to its complete absence. CDH may allow the stomach and intestines to move into the chest cavity, crowding the heart and lungs, which can lead to underdevelopment of the lungs (pulmonary hypoplasia) and potentially life-threatening complications. CDH can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic set of findings. Currently, about 15%-20% of individuals with CDH have an identifiable cause, such as a recognized chromosome abnormality or a single gene disorder, while the cause is unknown in the remaining 80%-

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.7293) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence of about 1 in 2,500 i...
2. `medquad_ans_0000173-1` (score=0.6996) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as touch, pain, heat, and ...
3. `medquad_ans_0000173-5` (score=0.6964) - These resources address the diagnosis or management of Charcot-Marie-Tooth disease: - Gene Review: Gene Review: Charcot-Marie-Tooth Hereditary Neuropathy Overview - Gene Review: Gene Review: Charcot-Marie-Tooth Neuropathy Type 1 - Gene R...
4. `medquad_ans_0001104-4` (score=0.6811) - How is Charcot-Marie-Tooth disease type 2F inherited? Charcot-Marie-Tooth disease type 2F is inherited in an autosomal dominant manner. This means that only one mutated copy of the gene in each cell is sufficient to cause the condition. ...
5. `medquad_ans_0001088-1__dup1` (score=0.6796) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common symptoms may includ...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Damage to these nerves can result in loss of sensation and wasting (atrophy) of muscles in the feet, legs, and hands. The disease typically becomes apparent in adolescence or early adulthood, though onset can occur from early childhood through late adulthood. Symptoms vary in severity, even among family members, and may include balance difficulties, clumsiness, muscle weakness, foot abnormalities such as high arches or flat feet, difficulty walking on the heel, and decreased sensitivity to touch, heat, and cold. In some cases, it may also cause hearing loss, deafness, or vision loss. There are several types, including Type 1 (CMT1), which involves abnormalities in myelin, and Type 2 (CMT2),

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Top docs**:

1. `medquad_ans_0000669-4` (score=0.7315) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is suf...
2. `medquad_ans_0000669-2` (score=0.6718) - The prevalence of monilethrix is unknown.
3. `medquad_ans_0004109-5` (score=0.6529) - Is genetic testing available for Mondini dysplasia? Genetic testing may be available for Mondini dysplasia if it is associated with a specific syndrome for which genetic testing is available, or if a mutation has previously been identifi...
4. `medquad_ans_0000669-1` (score=0.6512) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin word for necklace (m...
5. `medquad_ans_0000669-3` (score=0.6449) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for making proteins that gi...

**Answer**:

Yes, monilethrix can be inherited in different patterns depending on the cause. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. In rare cases, the condition results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000669-4].

**Citations detected**: `medquad_ans_0000669-4`
**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

## Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Top docs**:

1. `medquad_ans_0000215-4__dup2` (score=0.6997) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1
2. `medquad_ans_0000217-4__dup1` (score=0.6997) - In many cases, a child with VUR has no symptoms. When symptoms are present, the most common is a urinary tract infection (UTI). VUR can lead to infection because urine that remains in the childs urinary tract provides a place for bacteri...
3. `medquad_ans_0000217-5` (score=0.6809) - When a child with VUR gets a UTI, bacteria can move into the kidney and lead to scarring. Scarring of the kidney can be associated with high blood pressure and kidney failure. However, most children with VUR who get a UTI recover without...
4. `medquad_ans_0000223-12` (score=0.6788) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, the part of the dige...
5. `medquad_ans_0000169-1__dup3` (score=0.6707) - Pyelonephritis is a type of urinary tract infection (UTI) that affects one or both kidneys.

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microbes, typically bacteria that live in the bowel. These bacteria can enter the urinary tract and lead to infection. Most UTIs are not serious, but some can lead to serious complications such as pyelonephritis, which affects one or both kidneys. Symptoms may range from mild to severe or may be absent. UTIs are diagnosed through urine testing, and treatment involves antibiotics. Children with a UTI may be evaluated for conditions like vesicoureteral reflux (VUR) or urinary tract blockages. Girls are more likely than boys to develop UTIs. [medquad_ans_0000223-12] [medquad_ans_0000169-1__dup3]

**Citations detected**: `medquad_ans_0000223-12`, `medquad_ans_0000169-1__dup3`
**Supported citations**: `medquad_ans_0000223-12`, `medquad_ans_0000169-1__dup3`
**Unsupported citations**: _none_

## Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000804-2` (score=0.6597) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Autosomal dominant pol...
2. `medquad_ans_0000135-2__dup2` (score=0.6539) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems with blood vessels in ...
3. `medquad_ans_0000135-4__dup3` (score=0.6539) - Researchers do not fully understand what causes cysts to grow in the kidneys of people with CKD. The fact that these cysts occur only in the kidneys and not in other parts of the body, as in PKD, indicates that the processes that lead to...
4. `medquad_ans_0000135-1__dup4` (score=0.6486) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease that causes the kidneys...
5. `medquad_ans_0004939-1` (score=0.6466) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the their ability to fil...

**Answer**:

Complications of Polycystic Kidney Disease (PKD) include high blood pressure and problems with blood vessels in the brain and heart [medquad_ans_0000135-2__dup2].

**Citations detected**: `medquad_ans_0000135-2__dup2`
**Supported citations**: `medquad_ans_0000135-2__dup2`
**Unsupported citations**: _none_

## Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Top docs**:

1. `medquad_ans_0000123-6` (score=0.6723) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care provider may ask for...
2. `medquad_ans_0000649-1__dup1` (score=0.6589) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contaminated food or water....
3. `medquad_ans_0000096-2__dup2` (score=0.6255) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns disease. The most commo...
4. `medquad_ans_0000123-4__dup2` (score=0.6225) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 months old. Symptoms u...
5. `medquad_ans_0000391-1__dup1` (score=0.6178) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. Viral gastroenteritis...

**Answer**:

Norovirus infections are a type of viral gastroenteritis that cause inflammation of the stomach and intestines. Symptoms include diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, or body aches. The infection can spread from person to person, through contaminated food or water, or by touching a contaminated surface. It can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment involves bed rest and plenty of fluids to prevent dehydration. Proper hand washing and safe food preparation can help prevent infection [medquad_ans_0000649-1__dup1].

**Citations detected**: `medquad_ans_0000649-1__dup1`
**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

## Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0001497-3` (score=0.6950) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic set of findings. Curre...
2. `medquad_ans_0000222-4` (score=0.6929) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...
3. `medquad_ans_0000222-2` (score=0.6893) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
4. `medquad_ans_0000392-3` (score=0.6717) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chromosome regions have...
5. `medquad_ans_0000222-5` (score=0.6623) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia University Medical Center...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia (CDH) include chromosomal microdeletions (tiny loss of genetic material) or microduplications (an extra copy of genetic material), mutations in major genes important for diaphragm development, combined effects of multiple minor genetic mutations or variants (polygenic inheritance), and gene-environment interactions (multifactorial inheritance) [medquad_ans_0001497-3]. Additionally, duplications and deletions in several chromosome regions have been associated with CDH and some features of Fryns syndrome, although no specific genetic change has been identified that causes all signs and symptoms of Fryns syndrome [medquad_ans_0000392-3]. In some cases, CDH is linked to recognized chromosome abnormalities or single gene disorders, classifying it as syndromic CDH [medquad_ans_0001497-

**Citations detected**: `medquad_ans_0001497-3`, `medquad_ans_0000392-3`
**Supported citations**: `medquad_ans_0001497-3`, `medquad_ans_0000392-3`
**Unsupported citations**: _none_

## Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Top docs**:

1. `medquad_ans_0004534-1` (score=0.6944) - Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) characterized by erythroderma (skin redness), desquamation (peeling skin), alopecia (hair loss), chronic diarrhea, failure to thrive, lymphadenopath...
2. `medquad_ans_0004534-2` (score=0.6915) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCID). The characterist...
3. `medquad_ans_0004534-3` (score=0.6908) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 and RAG2 genes have ...
4. `medquad_ans_0001068-2` (score=0.6849) - ZAP70-related SCID is a rare disorder. Only about 20 affected individuals have been identified. The prevalence of SCID from all genetic causes combined is approximately 1 in 50,000.
5. `medquad_ans_0004534-4` (score=0.6809) - How might Omenn syndrome be treated? The standard treatment for Omenn syndrome is bone marrow transplantation or cord blood stem cell transplantation. General care for any patient with severe combined immunodeficiency (SCID), including O...

**Answer**:

Omenn syndrome is caused by mutations in the RAG1 or RAG2 genes. Additional causative genes have been identified [medquad_ans_0004534-1].

**Citations detected**: `medquad_ans_0004534-1`
**Supported citations**: `medquad_ans_0004534-1`
**Unsupported citations**: _none_

## Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Top docs**:

1. `medquad_ans_0000323-2` (score=0.6755) - Episodic ataxia is uncommon, affecting less than 1 in 100,000 people. Only types 1 and 2 have been identified in more than one family, and type 2 is by far the most common form of the condition.
2. `medquad_ans_0002439-4` (score=0.6721) - How is Friedreich ataxia inherited? Friedreich ataxia is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. The parents of an aff...
3. `medquad_ans_0000080-2` (score=0.6702) - Ataxia-telangiectasia occurs in 1 in 40,000 to 100,000 people worldwide.
4. `medquad_ans_0000032-3__dup1` (score=0.6607) - The prognosis for individuals with ataxia and cerebellar/spinocerebellar degeneration varies depending on its underlying cause.
5. `medquad_ans_0000097-1` (score=0.6571) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and symptoms of the disorder first appear in early to mid-adulthood. People with this condition initially experience impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. They may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features of ARCA1 include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive, often resulting in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Citations detected**: `medquad_ans_0000097-1`
**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

## Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Top docs**:

1. `medquad_ans_0000050-2__dup1` (score=0.7002) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
2. `medquad_ans_0000049-2__dup1` (score=0.7002) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
3. `medquad_ans_0000049-4__dup1` (score=0.6821) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
4. `medquad_ans_0000050-4__dup1` (score=0.6821) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
5. `medquad_ans_0000123-1__dup2` (score=0.6557) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed arm - Lack of muscl...

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000050-2__dup1].

**Citations detected**: `medquad_ans_0000050-2__dup1`
**Supported citations**: `medquad_ans_0000050-2__dup1`
**Unsupported citations**: _none_

## Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Top docs**:

1. `medquad_ans_0000216-6` (score=0.7011) - Treatment for urine blockage depends on the cause and severity of the blockage. Hydronephrosis discovered before the baby is born rarely requires immediate action, especially if it is only on one side. The condition often goes away witho...
2. `medquad_ans_0000141-9` (score=0.6982) - - An ectopic kidney is a birth defect in which a kidney is located below, above, or on the opposite side of its usual position. - Factors that may lead to an ectopic kidney include - poor development of a kidney bud - a defect in the kid...
3. `medquad_ans_0000216-4__dup1` (score=0.6978) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ultrasound of a baby...
4. `medquad_ans_0000216-7` (score=0.6968) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
5. `medquad_ans_0000216-3__dup1` (score=0.6837) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a closed urethra, which m...

**Answer**:

The provided context does not specify the exact causes of urine blockage in newborns. While it mentions that urine blockage depends on the cause and severity, and discusses conditions like ectopic kidney as a birth defect that may affect kidney position, it does not directly link ectopic kidney or any other factor to urine blockage in newborns. Therefore, based on the given information, the causes of urine blockage in newborns cannot be determined.

[medquad_ans_0000216-6]

**Citations detected**: `medquad_ans_0000216-6`
**Supported citations**: `medquad_ans_0000216-6`
**Unsupported citations**: _none_

## Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Top docs**:

1. `medquad_ans_0000526-4` (score=0.7218) - Intranuclear rod myopathy is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases are not inherited; they result from new mutations in the gene and occur i...
2. `medquad_ans_0000526-2` (score=0.7140) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
3. `medquad_ans_0000526-1` (score=0.6907) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor muscle tone (hypoto...
4. `medquad_ans_0000526-5` (score=0.6855) - These resources address the diagnosis or management of intranuclear rod myopathy: - Genetic Testing Registry: Nemaline myopathy 3 These resources from MedlinePlus offer information about the diagnosis and management of various health con...
5. `medquad_ans_0000526-3` (score=0.6798) - Intranuclear rod myopathy is caused by a mutation in the ACTA1 gene. This gene provides instructions for making a protein called skeletal alpha ()-actin, which is part of the actin protein family. Actin proteins are important for cell mo...

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown. [medquad_ans_0000526-2]

**Citations detected**: `medquad_ans_0000526-2`
**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

## Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Top docs**:

1. `medquad_ans_0000036-11` (score=0.7663) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure. Primary, or essent...
2. `medquad_ans_0000036-1__dup3` (score=0.7538) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. What Is Blood Pressure? Blood pressure is the force of blood pushing against the walls of the blood vessels as...
3. `medquad_ans_0000036-12` (score=0.7459) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hypertension. Pre-hype...
4. `medquad_ans_0000036-20` (score=0.7311) - High blood pressure is treated with lifestyle changes and medicines. Treatment can help control blood pressure, but it will not cure high blood pressure, even if your blood pressure readings appear normal. If you stop treatment, your blo...
5. `medquad_ans_0000036-21` (score=0.7243) - In most cases, the goal is probably to keep your blood pressure below 140/90 mmHg (130/80 if you have diabetes or chronic kidney disease). Normal blood pressure is less than 120/80. Ask your doctor what your blood pressure goal should be...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. It is defined as the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If blood pressure rises and stays high over time, it is called high blood pressure. High blood pressure is dangerous because it makes the heart work too hard and the high force of blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes [medquad_ans_0000036-1__dup3]. There are two main types: primary (or essential) high blood pressure, which is the most common type and develops over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or use of certain medicines and usually resolves after the cause is treated or removed [medquad_ans_0000036-11].

**Citations detected**: `medquad_ans_0000036-1__dup3`, `medquad_ans_0000036-11`
**Supported citations**: `medquad_ans_0000036-1__dup3`, `medquad_ans_0000036-11`
**Unsupported citations**: _none_

## Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Top docs**:

1. `medquad_ans_0001051-2` (score=0.6990) - X-linked dystonia-parkinsonism has been reported in more than 500 people of Filipino descent, although it is likely that many more Filipinos are affected. Most people with this condition can trace their mother's ancestry to the island of...
2. `medquad_ans_0001051-5` (score=0.6887) - These resources address the diagnosis or management of X-linked dystonia-parkinsonism: - Gene Review: Gene Review: X-Linked Dystonia-Parkinsonism Syndrome - Genetic Testing Registry: Dystonia 3, torsion, X-linked These resources from Med...
3. `medquad_ans_0001051-4` (score=0.6816) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
4. `medquad_ans_0001051-1` (score=0.6742) - X-linked dystonia-parkinsonism is a movement disorder that has been found only in people of Filipino descent. This condition affects men much more often than women. Parkinsonism is usually the first sign of X-linked dystonia-parkinsonism...
5. `medquad_ans_0000859-2` (score=0.6559) - Rapid-onset dystonia parkinsonism appears to be a rare disorder, although its prevalence is unknown. It has been diagnosed in individuals and families from the United States, Europe, and Korea.

**Answer**:

Yes, X-linked dystonia-parkinsonism is inherited in an X-linked recessive pattern [medquad_ans_0001051-4].

**Citations detected**: `medquad_ans_0001051-4`
**Supported citations**: `medquad_ans_0001051-4`
**Unsupported citations**: _none_

## Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Top docs**:

1. `medquad_ans_0000120-2` (score=0.7484) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
2. `medquad_ans_0000120-3` (score=0.6858) - Mutations in the ACAT1 gene cause beta-ketothiolase deficiency. This gene provides instructions for making an enzyme that is found in the energy-producing centers within cells (mitochondria). This enzyme plays an essential role in breaki...
3. `medquad_ans_0000120-1` (score=0.6841) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molec...
4. `medquad_ans_0000120-5` (score=0.6715) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer information about the ...
5. `medquad_ans_0000993-2` (score=0.6687) - Triosephosphate isomerase deficiency is likely a rare condition; approximately 40 cases have been reported in the scientific literature.

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process the protein building block (amino acid) called isoleucine. It also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. This condition is caused by mutations in the ACAT1 gene, which provides instructions for making an enzyme found in the mitochondria. The ACAT1 enzyme plays an essential role in breaking down proteins and fats; mutations in this gene reduce or eliminate the enzyme's activity, leading to a buildup of toxic compounds in the blood. These substances cause the blood to become too acidic (ketoacidosis), which can damage tissues and organs, especially the nervous system. Signs and symptoms typically appear between 6 months and 24 months of age and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.6396) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread...
2. `medquad_ans_0000006_5-9` (score=0.6165) - Key Points - There are different types of treatment for children with brain stem glioma. - Children with brain stem glioma should have their treatment planned by a team of health care providers who are experts in treating childhood brain...
3. `medquad_ans_0000006_3-9` (score=0.6029) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treating childhood brain ...
4. `medquad_ans_0000006_5-1` (score=0.6025) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - There are two types o...
5. `medquad_ans_0000006_6-8` (score=0.6021) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health care providers who ...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Citations detected**: `medquad_ans_0000006_5-7`
**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

## Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000266-3__dup1` (score=0.6835) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears to improve chances fo...
2. `medquad_ans_0000266-1__dup2` (score=0.6671) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal cord (conus medullari...
3. `medquad_ans_0000061-4__dup1` (score=0.6543) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research focuses on finding ...
4. `medquad_ans_0000054-4__dup1` (score=0.6444) - The NINDS supports and conducts a wide range of research on spinal cord disorders such as BSS. The goal of this research is to find ways to prevent, treat, and, ultimately, cure these disorders.
5. `medquad_ans_0000006_1-3` (score=0.6419) - The cause of most adult brain and spinal cord tumors is not known.

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or may develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course of the disorder. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. This delayed presentation is related to the degree of strain on the spinal cord over time and may be exacerbated by sports, pregnancy, or spinal column narrowing (stenosis) with age. T

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Top docs**:

1. `medquad_ans_0000156-2` (score=0.6818) - Carpenter syndrome is thought to be a rare condition; approximately 70 cases have been described in the scientific literature.
2. `medquad_ans_0000419-1__dup1` (score=0.6816) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel syndrome - compress...
3. `medquad_ans_0000058-4__dup1` (score=0.6743) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to conduct fundamental research on the brain and nervous system, and to use that knowledge to reduce the burden of neurological disease. NINDS-funded s...
4. `medquad_ans_0000058-3__dup1` (score=0.6540) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work, people can perform ...
5. `medquad_ans_0000703-1__dup1` (score=0.6496) - Your peripheral nerves are the ones outside your brain and spinal cord. Like static on a telephone line, peripheral nerve disorders distort or interrupt the messages between the brain and the rest of the body. There are more than 100 kin...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as it responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Preventive measures such as performing stretching exercises, taking frequent rest breaks, wearing splints to keep wrists straight, and maintaining correct posture and wrist position can help manage or prevent symptoms [medquad_ans_0000058-3__dup1].

**Citations detected**: `medquad_ans_0000058-3__dup1`
**Supported citations**: `medquad_ans_0000058-3__dup1`
**Unsupported citations**: _none_

## Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Top docs**:

1. `medquad_ans_0000296-2` (score=0.6980) - Although its prevalence is unknown, Donnai-Barrow syndrome appears to be a rare disorder. A few dozen affected individuals have been reported in many regions of the world.
2. `medquad_ans_0000296-4` (score=0.6854) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. In almost all cases, the parents of an individual with an autosomal recessive condition each carry one copy o...
3. `medquad_ans_0000296-5` (score=0.6732) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphragmatic Hernia - Medli...
4. `medquad_ans_0000130-2__dup1` (score=0.6730) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportive care.
5. `medquad_ans_0001901-1` (score=0.6717) - Donnai Barrow syndrome is an inherited disorder that affects many parts of the body. People with this condition generally have characteristic facial features, severe sensorineural hearing loss, vision problems and an absent or underdevel...

**Answer**:

Treatment for Donnai-Barrow syndrome is based on the signs and symptoms present in each person and may include hearing aids and/or cochlear implants for hearing loss, corrective lenses for vision problems, and surgery for certain physical abnormalities [medquad_ans_0000130-2__dup1]. Additionally, management may involve genetic counseling, diagnostic testing, and other supportive care as outlined in resources such as the Genetic Testing Registry and MedlinePlus Encyclopedia [medquad_ans_0000296-5].

**Citations detected**: `medquad_ans_0000130-2__dup1`, `medquad_ans_0000296-5`
**Supported citations**: `medquad_ans_0000130-2__dup1`, `medquad_ans_0000296-5`
**Unsupported citations**: _none_

## Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Top docs**:

1. `medquad_ans_0001134-1` (score=0.7117) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not conform to any of the reco...
2. `medquad_ans_0000189-4__dup2` (score=0.7061) - The exact rates of porphyria are unknown and vary around the world. For example, porphyria cutanea tarda is most common in the United States, and variegate porphyria is most common in South America.1
3. `medquad_ans_0000811-2` (score=0.7055) - The exact prevalence of porphyria is unknown, but it probably ranges from 1 in 500 to 1 in 50,000 people worldwide. Overall, porphyria cutanea tarda is the most common type of porphyria. For some forms of porphyria, the prevalence is unk...
4. `medquad_ans_0000189-3__dup2` (score=0.7034) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences: - Acute porphyrias ...
5. `medquad_ans_0000117-1__dup1` (score=0.6997) - Acute intermittent porphyria (AIP) is one of the liver (hepatic) porphyrias. AIP is caused by low levels of porphobilinogen deaminase (PBGD), an enzyme also often called hydroxymethylbilane synthase. The low levels of PBGD are generally ...

**Answer**:

Porphyria is a group of disorders characterized by abnormalities in the heme biosynthetic pathway, resulting in low levels of specific enzymes. Each of the eight types of porphyria corresponds to a deficiency of a specific enzyme in this pathway [medquad_ans_0000189-3__dup2]. Porphyrias are classified based on symptoms into acute or cutaneous types: acute porphyrias affect the nervous system and occur rapidly, while cutaneous porphyrias primarily affect the skin [medquad_ans_0000189-3__dup2]. Additionally, porphyrias are classified as erythropoietic or hepatic, depending on whether porphyrin overproduction occurs mainly in the bone marrow (erythropoietic) or the liver (hepatic) [medquad_ans_0000189-3__dup2]. Examples include acute intermittent porphy

**Citations detected**: `medquad_ans_0000189-3__dup2`
**Supported citations**: `medquad_ans_0000189-3__dup2`
**Unsupported citations**: _none_

## Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.6958) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable degrees of hearing lo...
2. `medquad_ans_0001900-1` (score=0.6888) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual loss and color visi...
3. `medquad_ans_0004557-1` (score=0.6764) - Optic atrophy 1 is a condition that mainly affects vision, but may include other features. Vision loss typically begins within the first decade of life; severity varies widely among affected people (from nearly normal vision to complete ...
4. `medquad_ans_0000588-5` (score=0.6573) - These resources address the diagnosis or management of Leber hereditary optic neuropathy: - Gene Review: Gene Review: Leber Hereditary Optic Neuropathy - Gene Review: Gene Review: Mitochondrial Disorders Overview - Genetic Testing Regist...
5. `medquad_ans_0000073-5` (score=0.6560) - These resources address the diagnosis or management of Arts syndrome: - Gene Review: Gene Review: Arts Syndrome - Genetic Testing Registry: Arts syndrome - MedlinePlus Encyclopedia: Hearing Loss - MedlinePlus Encyclopedia: Movement, Unco...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is inherited in an autosomal recessive manner [medquad_ans_0000585-1__dup1].

**Citations detected**: `medquad_ans_0000585-1__dup1`
**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

## Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.7113) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process in cells known as g...
2. `medquad_ans_0000794-2` (score=0.7109) - Phosphoglycerate kinase deficiency appears to be a rare disorder. About 30 families with affected members have been reported in the scientific literature.
3. `medquad_ans_0000794-4` (score=0.6814) - This condition is inherited in an X-linked recessive pattern. The PGK1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the gene in each cell is ...
4. `medquad_ans_0000795-2` (score=0.6799) - Phosphoglycerate mutase deficiency is a rare condition; about 15 affected people have been reported in the medical literature. Most affected individuals have been African American.
5. `medquad_ans_0000794-5` (score=0.6776) - These resources address the diagnosis or management of phosphoglycerate kinase deficiency: - Children Living with Inherited Metabolic Diseases (CLIMB) (UK): Phosphoglycerate Kinase Deficiency - Genetic Testing Registry: Deficiency of pho...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in glycolysis, a critical energy-producing process in cells. Mutations in the PGK1 gene reduce the activity of phosphoglycerate kinase, disrupting energy production and leading to cell damage or cell death [medquad_ans_0000794-3]. The condition is inherited in an X-linked recessive pattern, as the PGK1 gene is located on the X chromosome [medquad_ans_0000794-4]. In males, one altered copy of the gene is sufficient to cause the condition, while in females, mutations in both copies are typically required to cause the disorder, though females with one altered copy may exhibit some features such as anemia [medquad_ans_0000794

**Citations detected**: `medquad_ans_0000794-3`, `medquad_ans_0000794-4`
**Supported citations**: `medquad_ans_0000794-3`, `medquad_ans_0000794-4`
**Unsupported citations**: _none_

## Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Top docs**:

1. `medquad_ans_0000850-5` (score=0.6623) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagnosis and management o...
2. `medquad_ans_0000453-5` (score=0.6350) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodeficiency These resources...
3. `medquad_ans_0001065-5` (score=0.6255) - These resources address the diagnosis or management of X-linked thrombocytopenia: - Gene Review: Gene Review: WAS-Related Disorders - Genetic Testing Registry: Thrombocytopenia, X-linked - National Heart Lung and Blood Institute: How is ...
4. `medquad_ans_0000608-4` (score=0.6228) - Most cases of lung cancer are not related to inherited gene changes. These cancers are associated with somatic mutations that occur only in certain cells in the lung. When lung cancer is related to inherited gene changes, the cancer risk...
5. `medquad_ans_0000849-4` (score=0.6223) - Pulmonary arterial hypertension is usually sporadic, which means it occurs in individuals with no known family history of the disorder. These non-familial cases are described as idiopathic pulmonary arterial hypertension. About 20 percen...

**Answer**:

The provided context does not specify the genetic changes related to pulmonary veno-occlusive disease. While it mentions that resources are available for the diagnosis and management of pulmonary veno-occlusive disease, including genetic counseling and genetic testing, no details about specific genetic mutations or changes are given [medquad_ans_0000850-5]. Therefore, the genetic changes related to pulmonary veno-occlusive disease cannot be determined from the provided information.

**Citations detected**: `medquad_ans_0000850-5`
**Supported citations**: `medquad_ans_0000850-5`
**Unsupported citations**: _none_

## Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Top docs**:

1. `medquad_ans_0001006-2` (score=0.6473) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada where it occurs in ab...
2. `medquad_ans_0001006-1` (score=0.6454) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up in tissues and organs...
3. `medquad_ans_0006262-1` (score=0.6423) - Tyrosinemia type 3 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a deficiency of the enzyme 4-hydroxyphenylpyruvate dioxygenase, on...
4. `medquad_ans_0000001_4-3` (score=0.6381) - Most people with CML have a gene mutation (change) called the Philadelphia chromosome. Every cell in the body contains DNA (genetic material) that determines how the cell looks and acts. DNA is contained inside chromosomes. In CML, part ...
5. `medquad_ans_0006260-1` (score=0.6363) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate hydrolase, one of the...

**Answer**:

The provided context does not specify the exact genetic changes related to tyrosinemia. It mentions that there are three types of tyrosinemia, each distinguished by their symptoms and genetic cause [medquad_ans_0001006-1], and that tyrosinemia type I is the most severe form with signs beginning in the first few months of life [medquad_ans_0001006-1]. However, no details about the specific genetic mutations or changes are given in the context. Therefore, based on the available information, the genetic changes related to tyrosinemia cannot be determined.

Answer: The context does not provide information on the specific genetic changes related to tyrosinemia. [medquad_ans_0001006-1]

**Citations detected**: `medquad_ans_0001006-1`
**Supported citations**: `medquad_ans_0001006-1`
**Unsupported citations**: _none_

## Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Top docs**:

1. `medquad_ans_0000436-2` (score=0.7470) - Graves disease affects about 1 in 200 people. The disease occurs more often in women than in men, which may be related to hormonal factors. Graves disease is the most common cause of thyroid overactivity (hyperthyroidism) in the United S...
2. `medquad_ans_0002675-1` (score=0.7446) - Graves' disease is an autoimmune disorder that leads to overactivity of the thyroid gland (hyperthyroidism). It is caused by an abnormal immune system response that causes the thyroid gland to produce too much thyroid hormones. Graves di...
3. `medquad_ans_0000004-6__dup2` (score=0.7387) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease is often diagnosed a...
4. `medquad_ans_0000004-7__dup2` (score=0.7339) - Treatment for Graves disease can sometimes affect pregnancy. After treatment with surgery or radioactive iodine, TSI antibodies can still be present in the blood, even when thyroid levels are normal. If a pregnant woman has received eith...
5. `medquad_ans_0000436-5` (score=0.7114) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 - Genetic Testing Reg...

**Answer**:

The treatments for Graves' disease include radioiodine therapy, antithyroid drugs, and thyroid surgery. Radioiodine therapy is the most common treatment in the United States and involves taking radioactive iodine-131 by mouth, which is absorbed by the thyroid gland and gradually destroys thyroid cells. This treatment may require more than one round and can take several weeks or months to show results. It is not used in pregnant women or women who are breastfeeding due to potential harm to the fetus or passage to breast milk. Almost everyone who receives radioiodine therapy eventually develops hypothyroidism and will need to take synthetic thyroid hormone. Alternative treatments, such as prescription steroids, may help prevent complications like Graves' ophthalmopathy (GO). Graves' disease is often managed by an endocrinologist who specializes in hormone-secreting glands [medquad_ans_0002675-1]medquad_ans_00000

**Citations detected**: `medquad_ans_0002675-1`
**Supported citations**: `medquad_ans_0002675-1`
**Unsupported citations**: _none_

## Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Top docs**:

1. `medquad_ans_0000545-2` (score=0.7284) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, where approximately 1 in...
2. `medquad_ans_0000039-2__dup1` (score=0.7141) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be treated appropriately a...
3. `medquad_ans_0000039-1__dup2` (score=0.7095) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, or stumbling. Sympto...
4. `medquad_ans_0000039-3__dup1` (score=0.7040) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten disease is often fatal b...
5. `medquad_ans_0000545-3` (score=0.6951) - Most cases of juvenile Batten disease are caused by mutations in the CLN3 gene. This gene provides instructions for making a protein whose function is unknown. It is unclear how mutations in the CLN3 gene lead to the characteristic featu...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. These lipopigments accumulate within cellular structures called lysosomes, leading to progressive cell death, especially in brain neurons. This results in symptoms such as cognitive impairment, worsening seizures, progressive loss of sight and motor skills, vision loss, and intellectual decline. Early signs may include subtle changes in personality and behavior, slow learning, clumsiness, or stumbling. Vision loss is often an early sign, leading to suspicion during an eye exam. The disease is most commonly caused by mutations in the CLN3 gene, though a small percentage of cases are due to mutations in other genes involved in lysosomal function. Juvenile Batten disease is the most common type of Neuronal Ceroid Lipofuscinosis (NCL),

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Top docs**:

1. `medquad_ans_0002061-3` (score=0.6787) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This gene provides instruc...
2. `medquad_ans_0002061-5` (score=0.6772) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak muscle tone and delayed ...
3. `medquad_ans_0002061-4` (score=0.6613) - How is Ehlers-Danlos syndrome progeroid type inherited? Ehlers-Danlos syndrome progeroid type is inherited in an autosomal recessive pattern. This means that an individual must have two non-functional copies of the B4GALT7 gene to be aff...
4. `medquad_ans_0000313-5` (score=0.6483) - These resources address the diagnosis or management of Ehlers-Danlos syndrome: - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Classic Type - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Hypermobility Type - Gene Review: Gene Re...
5. `medquad_ans_0002061-1` (score=0.6398) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, hair, and skeletal s...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a rare genetic disorder caused by mutations in both copies of the B4GALT7 gene located on chromosome 5. This gene provides instructions for making an enzyme involved in the proper formation of collagen, the main protein in connective tissue. When the enzyme is not produced adequately, collagen is formed incorrectly, leading to weak connective tissue and the symptoms of the disorder [medquad_ans_0002061-3]. The condition is inherited in an autosomal recessive pattern, meaning an individual must inherit two non-functional copies of the B4GALT7 gene—one from each parent—to be affected. Carriers with only one non-functional copy do not typically show symptoms [medquad_ans_0002061-4]. Researchers are still studying the exact mechanism by which B4GALT7 gene mutations lead to the signs and symptoms of the disorder [

**Citations detected**: `medquad_ans_0002061-3`, `medquad_ans_0002061-4`
**Supported citations**: `medquad_ans_0002061-3`, `medquad_ans_0002061-4`
**Unsupported citations**: _none_

## Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7016) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
2. `medquad_ans_0000416-3` (score=0.6918) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
3. `medquad_ans_0000416-5` (score=0.6647) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlinePlus offer informatio...
4. `medquad_ans_0000442-3` (score=0.6560) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step production (synthesis)...
5. `medquad_ans_0002603-1` (score=0.6530) - Glutaric acidemia type I (GA1) is an inherited disorder in which the body can't process certain proteins properly. People with GA1 have inadequate levels of an enzyme needed to break down certain amino acids. These amino acids and their ...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of the metabolism of the amino acid histidine and in the production of one of the forms of the vitamin folate [medquad_ans_0000416-3].

**Citations detected**: `medquad_ans_0000416-3`
**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

## Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Top docs**:

1. `medquad_ans_0001048-2` (score=0.7157) - X-linked chondrodysplasia punctata 2 has been estimated to affect fewer than 1 in 400,000 newborns. However, the disorder may actually be more common than this estimate because it is likely underdiagnosed, particularly in females with mi...
2. `medquad_ans_0001047-2` (score=0.7015) - The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.
3. `medquad_ans_0001047-5` (score=0.6900) - These resources address the diagnosis or management of X-linked chondrodysplasia punctata 1: - Gene Review: Gene Review: Chondrodysplasia Punctata 1, X-Linked - Genetic Testing Registry: Chondrodysplasia punctata 1, X-linked recessive Th...
4. `medquad_ans_0001048-5` (score=0.6867) - These resources address the diagnosis or management of X-linked chondrodysplasia punctata 2: - Gene Review: Gene Review: Chondrodysplasia Punctata 2, X-Linked - Genetic Testing Registry: Chondrodysplasia punctata 2 X-linked dominant Thes...
5. `medquad_ans_0000876-5` (score=0.6577) - These resources address the diagnosis or management of rhizomelic chondrodysplasia punctata: - Gene Review: Gene Review: Rhizomelic Chondrodysplasia Punctata Type 1 - Genetic Testing Registry: Rhizomelic chondrodysplasia punctata type 1 ...

**Answer**:

The provided context does not include information about the symptoms of Chondrodysplasia punctata 1, X-linked recessive. [medquad_ans_0001047-5] mentions resources for diagnosis or management but does not specify symptoms. Therefore, the symptoms cannot be determined from the given context.

**Citations detected**: `medquad_ans_0001047-5`
**Supported citations**: `medquad_ans_0001047-5`
**Unsupported citations**: _none_

## Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Top docs**:

1. `medquad_ans_0000132-11` (score=0.6723) - - Whipple disease is a rare bacterial infection that primarily affects the small intestine. Left untreated, Whipple disease gets worse and is usually life threatening. - Bacteria called Tropheryma whipplei (T. whipplei) cause Whipple dis...
2. `medquad_ans_0000132-9` (score=0.6666) - Experts have not yet found a way to prevent Whipple disease.
3. `medquad_ans_0000132-6` (score=0.6654) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following complications in other ar...
4. `medquad_ans_0000132-10` (score=0.6597) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
5. `medquad_ans_0000165-4__dup1` (score=0.6511) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge of the brain and nervous system and to use that knowledge to reduce the burden of neurological disease. The NINDS support...

**Answer**:

The treatments for Whipple disease include intravenous (IV) antibiotics for the first 2 weeks of treatment, followed by long-term oral antibiotics prescribed by a healthcare provider. Most patients experience relief from symptoms within the first week or two. [medquad_ans_0000132-11]

**Citations detected**: `medquad_ans_0000132-11`
**Supported citations**: `medquad_ans_0000132-11`
**Unsupported citations**: _none_

## Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.7292) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-2__dup1` (score=0.7055) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0000042-3__dup2` (score=0.7045) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks after the initial onset ...
4. `medquad_ans_0000676-3__dup1` (score=0.7019) - What causes Bell's palsy?
5. `medquad_ans_0000086-1__dup2` (score=0.6565) - Bell's palsy is the most common cause of facial paralysis. It usually affects just one side of the face. Symptoms appear suddenly and are at their worst about 48 hours after they start. They can range from mild to severe and include - Tw...

**Answer**:

The provided context does not specify the treatments for Bell's palsy. While it details symptoms, causes, and prognosis, no information on specific treatments is included [medquad_ans_0000676-4__dup1]. Therefore, based on the given context, the treatments for Bell's palsy cannot be determined.

**Citations detected**: `medquad_ans_0000676-4__dup1`
**Supported citations**: `medquad_ans_0000676-4__dup1`
**Unsupported citations**: _none_
