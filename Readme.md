# Joblex

## Description
This repository builds a knowledgebase of organizational information descriptors.
For more detail refer to the SocInfo 2019 publication: 
JobLex: A Lexico-Semantic Knowledgebase of Occupational Information Descriptors: [PDF](https://smartech.gatech.edu/handle/1853/61818)

If useful, please cite this in your work as: 
Saha, K., Reddy, M. D., & De Choudhury, M. (2019). JobLex: A Lexico-Semantic Knowledgebase of Occupational Information Descriptors. SocInfo.

Bibtex: 
@inproceedings{saha2019joblex,
  title={JobLex: A Lexico-Semantic Knowledgebase of Occupational Information Descriptors},
  author={Saha, Koustuv and Reddy, Manikanta D and De Choudhury, Munmun},
  year = {2019},
  booktitle={SocInfo}
}

### Related Research:
Das Swain, V.*, Saha, K.*, Reddy, M. D., Rajvanshy, H., Abowd, G. D., & De Choudhury, M. (2020). Modeling Organizational Culture with Workplace Experiences Shared on Glassdoor. In Proceedings of the CHI Conference on Human Factors in Computing Systems, (* co-primary authors). [PDF](https://koustuv.com/papers/CHI20_OrganizationalCulture.pdf)

Saha, K., Reddy, M.D., Mattingly, S.M., Moskal, E., Sirigiri, A., & De Choudhury, M., (2019). LibRA: On LinkedIn based Role Ambiguity and Its Relationship with Wellbeing and Job Performance. In Proceedings of the ACM on Human-Computer Interaction (PACM HCI), 3, (CSCW), 137., Presented at CSCW 2019. [PDF](https://koustuv.com/papers/CSCW19_LibRA.pdf)

## Requirements

+ Python3, and associated libraries.

+ Download Content Model Reference from [ONET Center](https://www.onetcenter.org/database.html#individual-files)

+ Word embedding vector representations (Ex: [Glove Twitter](https://nlp.stanford.edu/projects/glove/))

## Generating Lexicon

    python Joblex.py

