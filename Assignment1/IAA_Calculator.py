'''
BMI 550: Applied BioNLP
Assignment 1: Inter-annotator agreement measurement

NOTES:
We will be doing a simplified version of agreement measurement.
We will use Cohen's kappa.

First, we load all the possible CUIs. Then, for each post, each CUI
will have 1 entry, and an additional entry for negation.

Following that, we compute the agreement between the binary arrays
indicating the presence or absence of each entry.

@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu


Created: 08/27/2023
'''
from collections import defaultdict
import pandas as pd
from sklearn.metrics import cohen_kappa_score

#Load all the cuis
#C0000000 is for anything labeled as 'other'
cuis = []
infile = open('./cuilist.txt')
for line in infile:
    cuis.append(line.strip())
#print (cuis)
#print (len(cuis))

#Now that we have the cuis loaded,
#we need to add negated or non-negated information.
#A simple way can be to append a '-0' or a '-1' tag indicating
#if a concept is negated or not.
cuis_with_neg_marker = []
for cui in cuis:
    cuis_with_neg_marker.append(cui+'-0')
    cuis_with_neg_marker.append(cui+'-1')

#print(cuis_with_neg_marker)

#Now we load the annotation files
def get_flagged_cuis_from_annotated_file (filepath):
    f1 = pd.read_excel(filepath)
    f1_flagged_cuis = defaultdict(list)
    for index,row in f1.iterrows():
        id_ = row['ID']
        cuis = row['Symptom CUIs'].split('$$$')
        neg_flags = row['Negation Flag'].split('$$$')
        for cui,flag in zip(cuis,neg_flags):
            if len(cui)>0 and len(flag)>0:
                f1_flagged_cuis[id_].append(cui+'-'+str(flag))
    return f1_flagged_cuis


f1_flagged_cuis = get_flagged_cuis_from_annotated_file('./data/annotation_files/sample_annot1.xlsx')
f2_flagged_cuis = get_flagged_cuis_from_annotated_file('./data/annotation_files/sample_annot_2.xlsx')
print(f1_flagged_cuis)
print(f2_flagged_cuis)


#Now we generate vectors for the computation...
#We only want to include IDs that are common in both files
#That is why we had stored them in dictionaries with IDs as the keys
commonids = list(set(f1_flagged_cuis.keys()).intersection(set(f2_flagged_cuis.keys())))
print(commonids)

f1_vec = []
f2_vec = []

for k in commonids:
    for c in cuis_with_neg_marker:
        if c in f1_flagged_cuis[k]:
            f1_vec.append(1)
        else:
            f1_vec.append(0)
        if c in f2_flagged_cuis[k]:
            f2_vec.append(1)
        else:
            f2_vec.append(0)

print (f1_vec)
print (f2_vec)
print(cohen_kappa_score(f1_vec,f2_vec))
