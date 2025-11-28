import numpy as np

'''
Precision = TP / (TP + FP)
TP (True Positives)
FP (False Positives)

'''
# par default 6 car (num_classes - 1) et il y a 5 labels (sans compter le background qui est le label 0)
def precision(modele_seg, ground_truth, labels):
    # modele_seg == segmentation par nnUNEt
    # ground_truth == etiquettes verite terrain (par le medecin)

    precisions = np.zeros(labels - 1) # Initialisation variable pour chaque etiquette/label (sauf background, donc -1)

    for etiquette in range(1, labels):  # Commencer a 1 car pas besoin du  background qui est le 0

        '''
        true positive (TP)
        false positive (FP)
        '''
        if etiquette == 6 :
            bladderneck = (modele_seg == 6) | (modele_seg== 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            TP = np.sum(bladderneck & verite_terrain)
            FP = np.sum(bladderneck & (~verite_terrain))
        elif etiquette == 7 :
            bladdertrigone = (modele_seg == 7) | (modele_seg == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            TP = np.sum(bladdertrigone &verite_terrain)
            FP = np.sum(bladdertrigone & (~verite_terrain))
        else :
            TP = np.sum((modele_seg == etiquette) & (ground_truth == etiquette))
            FP = np.sum((modele_seg == etiquette) & (ground_truth != etiquette))

        if (TP + FP) != 0 :
            precisions[etiquette - 1] = TP / (TP + FP)
        else :
            precisions[etiquette - 1] = 0
    
    return precisions
