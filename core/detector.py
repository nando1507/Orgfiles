import os
import urllib.request
import cv2
import numpy as np
from pathlib import Path

# URLs oficiais dos modelos ONNX do OpenCV Zoo
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

class FaceModels:
    """
    Gerencia o carregamento e download automático dos modelos YuNet (detecção) e SFace (reconhecimento).
    """
    def __init__(self, models_dir: str = "models", score_threshold: float = 0.7, nms_threshold: float = 0.3):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self.sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        self._ensure_models_downloaded()
        
        # Inicializa o detector com tamanho inicial (será ajustado dinamicamente para cada imagem)
        self.detector = cv2.FaceDetectorYN.create(
            str(self.yunet_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            5000
        )
        
        # Inicializa o reconhecedor SFace
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(self.sface_path),
            ""
        )

    def _ensure_models_downloaded(self):
        if not self.yunet_path.exists():
            print(f"[Modelos] Baixando YuNet para {self.yunet_path}...")
            urllib.request.urlretrieve(YUNET_URL, str(self.yunet_path))
        if not self.sface_path.exists():
            print(f"[Modelos] Baixando SFace para {self.sface_path}...")
            urllib.request.urlretrieve(SFACE_URL, str(self.sface_path))

    def detect_and_extract(self, image_bgr: np.ndarray, max_dim: int = 1280):
        """
        Detecta rostos na imagem e extrai as features (embeddings 128-d) e recortes de avatar.
        Retorna lista de tuplas: (embedding, face_crop_bgr, bounding_box, score)
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        scale = 1.0
        
        # Redimensiona para acelerar se a imagem for gigantesca
        work_img = image_bgr
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            work_w, work_h = int(w * scale), int(h * scale)
            work_img = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work_w, work_h = w, h

        self.detector.setInputSize((work_w, work_h))
        _, faces = self.detector.detect(work_img)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face_data in faces:
            # face_data: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            # Se redimensionamos, precisamos reescalonar as coordenadas do rosto de volta para a imagem original
            if scale != 1.0:
                face_orig = face_data.copy()
                face_orig[:14] = face_orig[:14] / scale
            else:
                face_orig = face_data

            try:
                # Alinhamento facial padrão SFace (112x112)
                aligned_face = self.recognizer.alignCrop(image_bgr, face_orig)
                # Extração do vetor de embedding 128-d normalizado
                feature = self.recognizer.feature(aligned_face)
                feature = feature.flatten()
                
                # Normalização L2 para garantir comparação cosseno exata
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm

                # Recorte do rosto para avatar de pré-visualização (com margem)
                x, y, fw, fh = int(face_orig[0]), int(face_orig[1]), int(face_orig[2]), int(face_orig[3])
                pad_x = int(fw * 0.2)
                pad_y = int(fh * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)
                
                crop = image_bgr[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else aligned_face
                
                results.append({
                    "embedding": feature,
                    "crop": crop,
                    "bbox": (x, y, fw, fh),
                    "score": float(face_orig[14])
                })
            except Exception as e:
                # Ignora falhas em detecções marginais
                continue

        return results

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calcula similaridade de cosseno entre dois embeddings normalizados."""
        return float(np.dot(feat1, feat2))
