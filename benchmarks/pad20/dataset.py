class PAD20:
    IMAGE_COLUMN = 'img_id'
    PATIENT_ID = 'patient_id'
    LESION_ID = 'lesion_id'
    RAW_CATEGORICAL_FEATURES = ["smoke", "drink", "background_father", "background_mother", "pesticide", "gender", 
            "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", 
            "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]

    LABELS = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    TARGET_COLUMN = "diagnostic"
    TARGET_NUMBER_COLUMN = "diagnostic_number"
    NUMERICAL_FEATURES = ['age', 'diameter_1', 'diameter_2']
    CATEGORICAL_FEATURES =  [
        'smoke_False', 'smoke_True',
        'drink_False', 'drink_True', 
        'background_father_AUSTRIA', 'background_father_BRASIL', 'background_father_BRAZIL',
        'background_father_CZECH', 'background_father_GERMANY',
        'background_father_ISRAEL', 'background_father_ITALY',
        'background_father_NETHERLANDS', 'background_father_POLAND',
        'background_father_POMERANIA', 'background_father_PORTUGAL',
        'background_father_SPAIN', 'background_father_UNK',
        'background_mother_BRAZIL', 'background_mother_FRANCE',
        'background_mother_GERMANY', 'background_mother_ITALY',
        'background_mother_NETHERLANDS', 'background_mother_NORWAY',
        'background_mother_POLAND', 'background_mother_POMERANIA',
        'background_mother_PORTUGAL', 'background_mother_SPAIN', 'background_mother_UNK', 
        'pesticide_False', 'pesticide_True',
        'gender_FEMALE', 'gender_MALE', 
        'skin_cancer_history_False','skin_cancer_history_True', 
        'cancer_history_False', 'cancer_history_True', 
        'has_piped_water_False', 'has_piped_water_True',
        'has_sewage_system_False', 'has_sewage_system_True', 
        'fitspatrick_1.0', 'fitspatrick_2.0', 'fitspatrick_3.0', 'fitspatrick_4.0', 'fitspatrick_5.0', 'fitspatrick_6.0',
        'region_ABDOMEN', 'region_ARM',
        'region_BACK', 'region_CHEST', 'region_EAR', 'region_FACE',
        'region_FOOT', 'region_FOREARM', 'region_HAND', 'region_LIP',
        'region_NECK', 'region_NOSE', 'region_SCALP', 'region_THIGH',
        'itch_False', 'itch_True', 'itch_UNK',
        'grew_False', 'grew_True', 'grew_UNK', 
        'hurt_False', 'hurt_True',  'hurt_UNK',
        'changed_False', 'changed_True', 'changed_UNK',
        'bleed_False', 'bleed_True', 'bleed_UNK',
        'elevation_False', 'elevation_True', 'elevation_UNK'
    ]