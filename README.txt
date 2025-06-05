Les différents notebook lié à la modélisation:


Préprocessing img Test --> Fichier qui a permis de faire des tests sur les images avant de modéliser. Débruitage, détection du roi, transformations
B2 Infiltration PA --> EfficientNet B2 Infiltration
5. V2 EfficientNet B0 PA et AP Atelectasis --> Efficient net B0 sur Atelctasis AP PA
Vit Atelectasis --> Vit sur les actelectasis et à la fin, il y a un modele vit sur du multi label
Atelectasis AP DensetNet121 --> Comme son nom l'indique, modèle DensetNet121 sur des Atelectasis AP

Pour les utiliser, récupérer les 3 csv dans le répertoire github et les importer dans Kaggle.
Il faut ensuite également importer dans chaque notebook le dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data



Le site se lance avec la commande: streamlit run app.py

