Le notebook lié à l'analyse exploratoire :

Importer le notebook analyse-exploratoire.ipynb sur kaggle
importer le dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data



Les différents notebook liés à la modélisation:

Préprocessing img Test --> Fichier qui a permis de faire des tests sur les images avant de modéliser. Débruitage, détection du roi, transformations
B2 Infiltration PA --> EfficientNet B2 Infiltration
5. V2 EfficientNet B0 PA et AP Atelectasis --> Efficient net B0 sur Atelctasis AP PA
Vit Atelectasis --> Vit sur les actelectasis et à la fin, il y a un modele vit sur du multi label
Atelectasis AP DensetNet121 --> Comme son nom l'indique, modèle DensetNet121 sur des Atelectasis AP

Pour les utiliser, récupérer les 3 csv dans le répertoire github et les importer dans Kaggle.
Il faut ensuite également importer dans chaque notebook le dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data



Le notebook lié à la segmentation :

Importer le notebook segmentation.ipynb sur kaggle

Importer les fichiers du dossier 'CSV Segmentation' : 

Mask_Mass.ZIP --> et le nommer "masse-mask" sur kaggle
Database_patients_normalis.csv --> et le nommer "dataset-nih-normalise" sur kaggle
Il faut ensuite également importer dans le notebook le dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data



Le site se lance avec la commande: streamlit run app.py
Il faut se placer dans Site pour lancer

