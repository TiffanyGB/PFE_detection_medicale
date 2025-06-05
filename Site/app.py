import streamlit as st
from PIL import Image
import torch
from torchvision.models import efficientnet_b0
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from modeles import GradCAM  # Assure-toi que GradCAM est bien défini dans modeles.py
import os
from timm import create_model
from torchvision.transforms.functional import to_pil_image
import io
import torch.nn.functional as F


def generate_vit_gradcam(model, input_tensor):
    # Récupère les features du dernier bloc
    def forward_hook(module, input, output):
        model.feature_map = output  # (B, num_tokens, dim)

    def backward_hook(module, grad_in, grad_out):
        model.gradients = grad_out[0]  # (B, num_tokens, dim)

    h_forward = model.blocks[-1].register_forward_hook(forward_hook)
    h_backward = model.blocks[-1].register_full_backward_hook(backward_hook)

    # 🔁 Forward + backward
    output = model(input_tensor)
    model.zero_grad()
    output.backward(torch.ones_like(output))

    grads = model.gradients  # (B, num_tokens, D)
    fmap = model.feature_map  # (B, num_tokens, D)

    # Compute weights by global average pooling
    weights = grads.mean(dim=1)  # (B, D)

    # Combine weights with features
    cam = torch.einsum('bd,bnd->bn', weights, fmap)  # (B, N)
    cam = cam[:, 1:]  # Enlève le token [CLS]
    cam = cam.reshape(1, 14, 14)  # Patch embeddings 14x14 pour vit_base_patch16_224
    cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()

    # Normalise
    cam -= cam.min()
    cam /= cam.max()
    return cam


st.set_page_config(layout="wide")  # Pour activer le mode large
st.title("Détection de maladies par image médicale 🩻")

# Initialisation session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# 1. Choix des pathologies
options = ['Atelectasis', 'Effusion', 'Infiltration', 'Mass',
           'Nodule', 'Pneumothorax', 'Consolidation']
selected_diseases = st.multiselect("Sélectionnez une ou plusieurs maladies à détecter", options)

# 2. Upload d'image (enregistrée en mémoire)
uploaded_image = st.file_uploader("Chargez une image", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    st.session_state.uploaded_image = uploaded_image

# 3. Définition du transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 4. Lancer l'analyse
if st.button("Lancer l'analyse") and st.session_state.uploaded_image and selected_diseases:
    image = Image.open(st.session_state.uploaded_image).convert("RGB")
    st.image(image, caption="Image chargée", width=300)
    input_tensor = transform(image).unsqueeze(0)

    # Scroll horizontal : wrapper avec container
    with st.container():
        st.write("EfficentNet-B0")
        cols = st.columns(len(selected_diseases))  # Une colonne par maladie

        for i, disease in enumerate(selected_diseases):
            with cols[i]:
                st.subheader(f"🩺 {disease}")
                model_path = f"modeles/efficientnet_{disease.lower()[0:2]}.pth"
                if not os.path.exists(model_path):
                    st.warning("Modèle manquant")
                    continue

                model = efficientnet_b0(num_classes=1)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()

                if prob > 0.5:
                    st.error(f"{disease} détecté\n({prob:.2f})")
                else:
                    st.success(f"Aucun signe\n({prob:.2f})")

                # Grad-CAM
                target_layer = model.features[-1]
                grad_cam = GradCAM(model, target_layer)
                cam = grad_cam.generate_cam(input_tensor)

                original_np = np.array(image)
                cam_resized = cv2.resize(cam, (original_np.shape[1], original_np.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                overlay = np.float32(heatmap) / 255 + np.float32(original_np) / 255
                overlay = overlay / np.max(overlay)

                st.image(overlay, caption=f"Grad-CAM : {disease}")
                
    # 5. Si Atelectasis est sélectionné, afficher les résultats avec ViT
    if "Atelectasis" in selected_diseases:
        st.markdown("---")
        st.subheader("🔬 ViT – Analyse Atelectasis")

        model_vit = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=1
        )

        vit_path = "/home/cytech/pfe_site_streamlit/modeles/vit_at_v2.pth"
        if not os.path.exists(vit_path):
            st.warning("Modèle ViT introuvable.")
        else:
            state_dict = torch.load(vit_path, map_location="cpu")
            model_vit.load_state_dict(state_dict)
            model_vit.eval()

            # Inférence
            with torch.no_grad():
                output_vit = model_vit(input_tensor)
                prob_vit = torch.sigmoid(output_vit).item()

            if prob_vit > 0.5:
                st.error(f"Atelectasis détecté par ViT\n({prob_vit:.2f})")
            else:
                st.success(f"Aucun signe d'Atelectasis selon ViT\n({prob_vit:.2f})")

        # 🎯 Génère la carte Grad-CAM
        cam = generate_vit_gradcam(model_vit, input_tensor)

        # Superposition sur l’image
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.axis('off')

        # Affichage Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        st.image(buf, caption="Interprétation Grad-CAM ViT", width=300)
        buf.close()


elif not selected_diseases:
    st.info("📋 Veuillez sélectionner au moins une maladie avant de lancer l'analyse.")
