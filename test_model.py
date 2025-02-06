import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from fastPET_UNet2D import UNet2D
from fastPET_UNet3D import UNet3D
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def get_device():
    """
    Détermine le meilleur device disponible
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def load_checkpoint(checkpoint_path, model, device='mps'):
    """
    Charge uniquement les poids du modèle depuis un checkpoint d'entraînement
    """
    if device == 'mps':
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def process_image(image_path, target_size=(480, 480)):
    """
    Charge et prétraite une image pour le modèle
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img).astype(float) / 255.0
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img_array

def denoise_image(img_tensor, model, device='mps'):
    """
    Applique le débruitage avec le modèle
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        denoised = model(img_tensor)
    
    denoised = denoised.cpu().numpy()
    denoised = np.squeeze(denoised)
    denoised = np.clip(denoised, 0, 1)
    
    return denoised

def add_comparison_to_pdf(pdf, original, denoised, title):
    """
    Ajoute une comparaison avant/après à un PDF existant
    """
    plt.figure(figsize=(12, 6))
    
    plt.suptitle(title, fontsize=12, y=0.98)
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Image bruitée')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Image débruitée')
    plt.axis('off')
    
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    MODEL_NAME = "checkpoint_epoch_036.pt"
    MODEL_PATH = "/Users/alex/Documents/GitHub/UNET4denoising/checkpoints/" + MODEL_NAME
    INPUT_DIR = "/Users/alex/Downloads/output_images"
    OUTPUT_DIR = "Results"
    
    DEVICE = get_device()
    print(f"Utilisation de : {DEVICE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(OUTPUT_DIR, f'denoising_results_{MODEL_NAME}_{timestamp}.pdf')
    
    try:
        # Initialiser le modèle et charger les poids
        model = UNet2D()
        model = load_checkpoint(MODEL_PATH, model, device=DEVICE)
        print("Modèle chargé avec succès")
        
        with PdfPages(pdf_path) as pdf:
            # Page de titre
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, 'Résultats de Débruitage\n\n' + timestamp, 
                    ha='center', va='center', fontsize=16)
            pdf.savefig()
            plt.close()
            
            # Traitement des images
            image_files = sorted([f for f in os.listdir(INPUT_DIR) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            for i, filename in enumerate(image_files, 1):
                print(f"Traitement de l'image {i}/{len(image_files)}: {filename}")
                
                image_path = os.path.join(INPUT_DIR, filename)
                
                try:
                    img_tensor, original = process_image(image_path)
                    denoised = denoise_image(img_tensor, model, device=DEVICE)
                    add_comparison_to_pdf(pdf, original, denoised, 
                                        f"Image {i}/{len(image_files)}: {filename}")
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de {filename}: {str(e)}")
                    continue
            
            # Page de métadonnées
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            metadata_text = (
                f"Métadonnées du traitement:\n\n"
                f"Date: {timestamp}\n"
                f"Nombre d'images traitées: {len(image_files)}\n"
                f"Device utilisé: {DEVICE}\n"
                f"Modèle: {os.path.basename(MODEL_PATH)}"
            )
            plt.text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10)
            pdf.savefig()
            plt.close()
        
        print(f"\nTraitement terminé. Résultats sauvegardés dans:\n{pdf_path}")
            
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")
        return

if __name__ == "__main__":
    main()