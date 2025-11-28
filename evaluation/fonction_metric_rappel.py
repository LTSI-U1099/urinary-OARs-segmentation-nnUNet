import numpy as np

# par default 6 car (num_classes - 1) et il y a 5 labels (sans compter le background qui est le label 0)
def recall(modele_seg, ground_truth, labels):
    # modele_seg == segmentation par nnUNEt
    # ground_truth == verite terrain (par le medecin)
    
    # Initialisation variable pour chaque etiquette/label (sauf background, donc -1)
    recall = np.zeros(labels - 1)

    for etiquette in range(1, labels):   # Commencer a 1 car pas besoin du  background qui est le 0

        '''
        Rappel = TP / (TP + FN)
        true positive (TP)
        false negative (FN)
        '''

        if etiquette == 6 :
            bladderneck = (modele_seg == 6) | (modele_seg== 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            TP = np.sum(bladderneck & verite_terrain)
            FN = np.sum((~bladderneck) & verite_terrain)
        elif etiquette == 7 :
            bladdertrigone = (modele_seg == 7) | (modele_seg == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            TP = np.sum(bladdertrigone & verite_terrain)
            FN = np.sum((~bladdertrigone) & verite_terrain)
        else :
            TP = np.sum((modele_seg == etiquette) & (ground_truth == etiquette))
            FN = np.sum((modele_seg != etiquette) & (ground_truth == etiquette))
        
        
        if TP + FN != 0:
            recall[etiquette - 1] = TP / (TP + FN)
        else:
            recall[etiquette - 1] = 0  # S'il n'y a pas de true positives et false negatives
    
    return recall
