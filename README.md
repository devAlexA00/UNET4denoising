# U-Net 4 denoising
Utilisation du repositorie Github de luisacbscs proposant déjà une implémentation 2D et 3D de UNet apprendre à débruiter des images. L'objectif est de pouvoir utilisé un modèle entraîné sur des images médicales présentant du bruit speckle, tel que des echographies. Le repositorie est utilisé comme une base qui a été modifié et adapté afin de pouvoir permettre un entrainement par morceau (checkpoint toute les 10 epochs). 

### Utilisation
Afin d'utiliser le code correctement, veuillez penser à changer le chemin dans Configuration dans le main de create_dataset.py, mais aussi dans le fichier fastPET_UNet_training.py et dans test_model.py. Les paramètres n'ont pas été changé depuis l'entrainement avec des images 480x480. Je recommande de passer à une résolution plus petite (car plus rapide et plus stable), ou alors de changer d'autres hyperparamètres pour essayer d'éviter le gradient vanishing observé avec la taille 480x480.

1. Définir un dataset adapté en utilisant create_dataset.py
2. Entrainer le modèle (ne pas oublier de récupérer les checkpoints)
3. Tester le dernier checkpoint produit avec test_model.py

Nous n'utilisons pas makeTorchDataset.py (un des fichiers originels)

Les résultats sont fournit. Si nécessaire, je peux fournir aussi des checkpoints à votre demande.