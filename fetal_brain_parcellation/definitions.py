# LABELS CODES
# OURS
LABELS = {'background': 0,
          'wm': 1,
          'ventricules': 2,  # subset of CSF
          'cerebellum': 3,
          'other_brain': 4}

# BOSTON fetal brain atlas
LABELS_BOSTON = {'Hippocampus_L': 37,
                 'Hippocampus_R': 38,
                 'Amygdala_L': 41,
                 'Amygdala': 42,
                 'Caudate_L': 71,
                 'Caudate_R': 72,
                 'Putamen_L': 73,
                 'Putamem_R': 74,
                 'Thalamus_L': 77,
                 'Thalamus_R': 78,
                 'CorpusCallosum': 91,
                 'Lateral_Ventricule_L': 92,
                 'Lateral_Ventricule_R': 93,
                 'Midbrain_L': 94,  # Brainstem in 'tissue' seg
                 'Midbrain_R': 95,
                 'Cerebellum_L': 100,
                 'Cerebellum_R': 101,
                 'Subthalamic_Nuc_L': 108,
                 'Subthalamic_Nuc_R': 109,
                 'Hippocampal_Comm': 110,
                 'Fornix': 111,
                 'Cortical_Plate_L': 112,
                 'Cortical_Plate_R': 113,
                 'Subplate_L': 114,
                 'Subplate_R': 115,
                 'Inter_Zone_L': 116,
                 'Inter_Zone_R': 117,
                 'Vent_Zone_L': 118,
                 'Vent_Zone_R': 119,
                 'White_Matter_L': 120,
                 'White_Matter_R': 121,
                 'Internal_Capsule_L': 122,
                 'Internal_Capsule_R': 123,
                 'CSF': 124,
                 'Misc': 125
                 }

# Conversion Boston labels to ours (for 'tissue' segmentations)
BOSTON2OURS = {37: 4,  # other
               38: 4,
               41: 4,
               42: 4,
               71: 4,
               72: 4,
               73: 4,  # bigger than what Nada would have segmented
               74: 4,  # bigger than what Nada would have segmented
               77: 4,
               78: 4,
               91: 1,  # WM
               92: 2,  # Ventricules
               93: 2,
               94: 4,
               95: 4,
               100: 3,  # cerebellum
               101: 3,
               108: 4,
               109: 4,
               110: 1,
               111: 1,
               112: 4,
               113: 4,
               114: 4,
               115: 4,
               116: 1,
               117: 1,
               118: 1,
               119: 1,
               120: 1,
               121: 1,
               122: 1,
               123: 1,
               124: 4,
               125: 4
               }

