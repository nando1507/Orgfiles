# Pacote core do organizador de fotos e vídeos por similaridade facial
from core.detector import FaceModels
from core.processor import MediaProcessor
from core.cluster import FaceClusterer, group_by_min_photos
from core.organizer import FileOrganizer
