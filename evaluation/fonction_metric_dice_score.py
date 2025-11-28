import numpy as np

'''
Dice Score = (2*TP) / (2*TP + FP+ FN)

evalue la similariter entre deux ensembles de pixels segmenter
score Dice = 0 : signifie qu'il n'y a aucune superposition entre le modele segmenter par nnUNEt et les etiquettes de verite terrain

    # modele_seg == segmentation par nnUNEt
    # ground_truth == etiquettes verite terrain (par le medecin)

'''

# par default 6 car (num_classes - 1) et il y a 5 labels (sans compter le background qui est le label 0)
def dice_score(segmentation_pred, ground_truth, labels):
    # modele_seg == segmentation par nnUNEt
    # ground_truth == etiquettes verite terrain (par le medecin)

    # Initialisation Dice Scores pour chaque etiquette/label (sauf background, donc -1)
    dice = np.zeros(labels - 1)
    
    
    for etiquette in range(1, labels):  # Commencer a 1 car pas besoin du  background qui est le 0
        
        '''
        true positive (TP)
        false positive (FP)
        false negative (FN)
        '''
        if etiquette == 6 :
            bladderneck = (segmentation_pred == 6) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            TP = np.sum(bladderneck & verite_terrain)
            FP = np.sum(bladderneck & (~verite_terrain))
            FN = np.sum((~bladderneck) & verite_terrain)
        elif etiquette == 7 :
            bladdertrigone = (segmentation_pred == 7) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            TP = np.sum(bladdertrigone & verite_terrain)
            FP = np.sum(bladdertrigone & (~verite_terrain))
            FN = np.sum((~bladdertrigone) & verite_terrain)
        else :
            TP = np.sum((segmentation_pred == etiquette) & (ground_truth == etiquette))
            FP = np.sum((segmentation_pred == etiquette) & (ground_truth != etiquette))
            FN = np.sum((segmentation_pred != etiquette) & (ground_truth == etiquette))
        
        if (TP + FP + FN) != 0 :
            # Calcul du Dice Score normaliser pour chaque label
            dice[etiquette - 1] = (2.0 * TP) / (2.0 * TP + FP + FN)
            # multiplier par 2 pour avoir sous la forme de normalisation, de 0 a 1
        else :
            dice[etiquette - 1] = 0
    
    return dice
