import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from core.detector import FaceModels

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def get_file_size_mb(path: Path) -> float:
    """Retorna o tamanho do arquivo em Megabytes (MB)."""
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0

def is_file_too_large(path: Path, max_size_mb: float = 50.0) -> bool:
    """Verifica se o arquivo excede o tamanho limite em MB."""
    if max_size_mb is None or max_size_mb <= 0:
        return False
    return get_file_size_mb(path) > max_size_mb

def read_image_safe(path: Path) -> Optional[np.ndarray]:
    """Lê imagem de forma segura no Windows, mesmo com acentos e caminhos com espaços."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

class MediaProcessor:
    def __init__(
        self,
        models: FaceModels,
        video_sample_fps: float = 1.0,
        internal_video_dedup_thresh: float = 0.50,
        max_file_size_mb: Optional[float] = 50.0,
        media_type: str = "both"
    ):
        self.models = models
        self.video_sample_fps = video_sample_fps
        self.internal_dedup_thresh = internal_video_dedup_thresh
        self.max_file_size_mb = max_file_size_mb
        self.media_type = media_type.lower()

    @staticmethod
    def list_media_files(directory: Path, media_type: str = "both", recursive: bool = True) -> List[Path]:
        """
        Lista arquivos de mídia no diretório conforme o filtro:
        media_type: 'both' / 'ambos', 'images' / 'imagens' / 'fotos', 'videos'.
        """
        mt = str(media_type).lower()
        include_images = mt in {"both", "ambos", "all", "images", "imagens", "fotos", "image", "foto"}
        include_videos = mt in {"both", "ambos", "all", "videos", "video", "vídeos", "vídeo"}

        valid_extensions = set()
        if include_images:
            valid_extensions.update(IMAGE_EXTENSIONS)
        if include_videos:
            valid_extensions.update(VIDEO_EXTENSIONS)

        files = []
        pattern = "**/*" if recursive else "*"
        for p in directory.glob(pattern):
            if p.is_file():
                ext = p.suffix.lower()
                if ext in valid_extensions:
                    files.append(p)
        return sorted(files)

    def process_image(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extrai faces de uma foto."""
        img = read_image_safe(file_path)
        if img is None:
            return []
        
        detections = self.models.detect_and_extract(img)
        faces = []
        for det in detections:
            faces.append({
                "file_path": file_path,
                "media_type": "image",
                "embedding": det["embedding"],
                "crop": det["crop"],
                "score": det["score"]
            })
        return faces

    def process_video(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extrai faces de um vídeo por amostragem de frames e agrupa internamente
        para não gerar centenas de embeddings repetidos do mesmo vídeo.
        """
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        # Calcula o intervalo de frames a pular
        step = max(1, int(fps / max(0.1, self.video_sample_fps)))
        
        frame_idx = 0
        raw_faces = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                detections = self.models.detect_and_extract(frame)
                for det in detections:
                    raw_faces.append({
                        "file_path": file_path,
                        "media_type": "video",
                        "embedding": det["embedding"],
                        "crop": det["crop"],
                        "score": det["score"]
                    })

            frame_idx += 1

        cap.release()

        if not raw_faces:
            return []

        # Deduplicação interna das faces detectadas no mesmo vídeo
        unique_video_faces: List[Dict[str, Any]] = []
        for face in raw_faces:
            matched = False
            for u in unique_video_faces:
                sim = FaceModels.cosine_similarity(face["embedding"], u["embedding"])
                if sim >= self.internal_dedup_thresh:
                    # Se for a mesma pessoa dentro do vídeo, atualiza se este recorte tiver maior qualidade/score
                    if face["score"] > u["score"]:
                        u["crop"] = face["crop"]
                        u["score"] = face["score"]
                    matched = True
                    break
            if not matched:
                unique_video_faces.append(face)

        return unique_video_faces

    def scan_directory(
        self,
        files: List[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Varre todos os arquivos fornecidos e retorna um dicionário com:
        - all_faces: lista de todas as faces detectadas
        - files_without_faces: lista de arquivos onde nenhum rosto foi encontrado
        - skipped_large_files: lista de arquivos ignorados por excederem o tamanho limite
        - file_face_map: mapeamento de arquivo -> lista de índices de faces
        """
        all_faces: List[Dict[str, Any]] = []
        files_without_faces: List[Path] = []
        skipped_large_files: List[Path] = []
        file_face_map: Dict[Path, List[int]] = {}

        total = len(files)
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)

            # Verifica se o arquivo é maior que o limite configurado (ex: 50MB)
            if is_file_too_large(file_path, self.max_file_size_mb):
                skipped_large_files.append(file_path)
                continue

            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                faces = self.process_image(file_path)
            elif ext in VIDEO_EXTENSIONS:
                faces = self.process_video(file_path)
            else:
                faces = []

            if not faces:
                files_without_faces.append(file_path)
            else:
                start_idx = len(all_faces)
                for f in faces:
                    all_faces.append(f)
                file_face_map[file_path] = list(range(start_idx, len(all_faces)))

        return {
            "all_faces": all_faces,
            "files_without_faces": files_without_faces,
            "skipped_large_files": skipped_large_files,
            "file_face_map": file_face_map
        }
