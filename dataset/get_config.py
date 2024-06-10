def get_config(config):

    
    if config['task_name'] == 'BBBP'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        config['dataset']['target'] = ["p_np"]


    elif config['task_name'] == 'Tox21'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        config['dataset']['target'] = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        config['dataset']['target'] = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        config['dataset']['target'] = ["HIV_active"]

    elif config['task_name'] == 'BACE'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        config['dataset']['target'] = ["Class"]

    elif config['task_name'] == 'SIDER'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        config['dataset']['target'] = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV.lower()':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        config['dataset']['target'] = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        config['dataset']['target'] = ["expt"]
    
    elif config["task_name"] == 'ESOL'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        config['dataset']['target'] = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        config['dataset']['target'] = ["exp"]
    
    elif config["task_name"] == 'qm7'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        config['dataset']['target'] = ["u0_atom"]

    elif config["task_name"] == 'qm8'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        config['dataset']['target'] = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9'.lower():
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        config['dataset']['target'] = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    elif config["task_name"] == '3mr'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/3MR/toy_label_mw350.csv'
        config['dataset']['target'] = ['label_full']

    elif config["task_name"] == 'benzene'.lower():
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/Benzene/benzene_smiles.csv'
        config['dataset']['target'] = ['label']


    else:
        
        raise ValueError('Undefined downstream task!')
    
    return config 