# Football Pitch Pose Estimation & 2D Tactical Radar

Système de vision par ordinateur pour la détection et le suivi des points clés d'un terrain de football (39 Keypoints), calcul d'homographie 2D temps réel, et projection tactique.

## 🚀 Fonctionnalités
- **Pose Estimation 39 Keypoints** : Modèle U-Net avec encodeur ResNet18 pré-entraîné et tête de visibilité (Confidence/Visibility Head).
- **Homographie & Projection 2D** : Calcul de la matrice d'homographie ($H$) via RANSAC pour projeter le terrain filmé sur un modèle 2D vue du dessus (Tactical Radar).
- **Accélération OpenVINO & ONNX** : Export en FP16 optimisé pour CPU/iGPU/NPU Intel (Intel Core Ultra).
- **Pipeline Vidéo** : Traitement et annotation vidéo en temps réel.

## 📂 Structure du projet
- `pitch_pose_ai.py` : Entraînement du modèle PyTorch (ResNet18 U-Net + Visibility Head).
- `pitch_homography_2d.py` : Calcul de l'homographie et génération du radar 2D tactique.
- `test_video_pose.py` : Pipeline d'inférence et d'annotation sur vidéo.
- `export_and_benchmark_ov.py` : Export ONNX & OpenVINO FP16 avec benchmark.
- `remap_dataset2.py` / `unify_datasets_39pts.py` : Scripts d'unification et de mapping des datasets de keypoints.
