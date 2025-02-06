import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def setup_directories(base_path):
    """Crée les répertoires nécessaires pour le dataset."""
    target_dir = os.path.join(base_path, 'Target')
    training_dir = os.path.join(base_path, 'Training')
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    return target_dir, training_dir

def check_image_resolution(image_path, min_size):
    """
    Vérifie si l'image a une résolution suffisante.
    
    Args:
        image_path (str): Chemin de l'image
        min_size (tuple): Taille minimale requise (width, height)
        
    Returns:
        bool: True si l'image a une résolution suffisante, False sinon
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # L'image doit avoir au moins la taille minimale dans sa plus petite dimension
            return min(width, height) >= min(min_size)
    except Exception:
        return False

def get_image_files(input_dir, skip_interval=1, min_size=(512, 512)):
    """
    Récupère les chemins des images en respectant l'intervalle de saut et la résolution minimale.
    
    Args:
        input_dir (str): Répertoire d'entrée
        skip_interval (int): Nombre d'images à sauter entre chaque sélection
        min_size (tuple): Taille minimale requise (width, height)
    """
    image_files = []
    count = 0
    total_processed = 0
    skipped_resolution = 0
    
    # Récupérer tous les fichiers images
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Vérifier la résolution
                if not check_image_resolution(file_path, min_size):
                    skipped_resolution += 1
                    continue
                
                total_processed += 1
                if total_processed % skip_interval == 1:  # Prend une image tous les skip_interval
                    image_files.append(file_path)
                    count += 1
    
    print(f"Images ignorées pour résolution insuffisante: {skipped_resolution}")
    print(f"Images sélectionnées: {len(image_files)} sur {total_processed} images valides")
    
    return sorted(image_files)  # Tri pour assurer un ordre consistant

def process_image(image_path, target_size=(512, 512)):
    """
    Charge, convertit en niveaux de gris et redimensionne une image.
    """
    with Image.open(image_path).convert('L') as img:
        # Récupérer les dimensions
        width, height = img.size
        
        # Calculer la plus petite dimension pour le crop carré
        crop_size = min(width, height)
        
        # Calculer les coordonnées pour le crop central
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Crop et redimensionnement
        img = img.crop((left, top, right, bottom))
        img = img.resize(target_size, Image.LANCZOS)
        
        return img

def add_speckle_noise(image_array, noise_level):
    """Ajoute du bruit speckle à une image."""
    # Convertir en float32 pour les calculs
    image_float = image_array.astype(np.float32) / 255.0
    
    # Générer le bruit
    noise = np.random.normal(0, noise_level, image_array.shape).astype(np.float32)
    
    # Ajouter le bruit multiplicatif
    noisy = image_float + image_float * noise
    
    # Normaliser entre 0 et 1
    noisy = np.clip(noisy, 0, 1)
    
    # Reconvertir en uint8
    return (noisy * 255).astype(np.uint8)

def process_dataset(input_dir, output_base_dir, skip_interval=1, target_size=(512, 512), noise_levels=np.linspace(0, 0.5, 11)):
    """
    Traite toutes les images du dataset et crée la structure de fichiers attendue.
    
    Args:
        input_dir (str): Répertoire contenant les images sources
        output_base_dir (str): Répertoire de sortie pour le dataset
        skip_interval (int): Nombre d'images à sauter entre chaque sélection
        target_size (tuple): Taille cible des images (width, height)
        noise_levels (np.array): Niveaux de bruit à appliquer
    """
    # Créer les répertoires de sortie
    target_dir, training_dir = setup_directories(output_base_dir)
    
    # Récupérer les fichiers images avec l'intervalle de saut et vérification de la résolution
    image_files = get_image_files(input_dir, skip_interval, target_size)
    
    print(f"Début du traitement des images...")
    
    # Traiter chaque image
    for idx, image_path in enumerate(tqdm(image_files), 1):
        try:
            # Créer le nom formaté de l'image
            formatted_name = f"{idx:04d}"
            
            # Traiter l'image originale
            processed_img = process_image(image_path, target_size)
            
            # Sauvegarder l'image cible
            target_path = os.path.join(target_dir, f"{formatted_name}.png")
            processed_img.save(target_path)
            
            # Convertir en array pour le traitement du bruit
            img_array = np.array(processed_img)
            
            # Créer les versions bruitées
            for noise_level in noise_levels:
                # Générer l'image bruitée
                noisy_img = add_speckle_noise(img_array, noise_level)
                
                # Créer le nom avec le niveau de bruit
                noise_str = f"{noise_level:.3f}".replace(".", "p")
                noisy_path = os.path.join(training_dir, f"{formatted_name}_{noise_str}.png")
                
                # Sauvegarder l'image bruitée
                Image.fromarray(noisy_img).save(noisy_path)
                
        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {str(e)}")
            continue

if __name__ == "__main__":
    # Configuration
    input_directory = "/Users/alex/Downloads/archive-5/animals/animals" # Penser à changer le chemin
    output_directory = "Dataset"
    skip_interval = 2 
    target_size = (512, 512)  # Taille cible des images
    noise_levels = np.linspace(0, 0.5, 11)  # 11 niveaux de bruit de 0 à 0.5
    
    # Traiter le dataset
    process_dataset(input_directory, output_directory, skip_interval, target_size, noise_levels)