import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.cluster import AgglomerativeClustering

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def group_by_min_photos(
    people: List[Dict[str, Any]],
    min_photos: int = 3,
    files_without_faces: Optional[List[Path]] = None,
    all_faces: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Agrupa mídias com base em uma quantidade mínima de fotos requerida.
    
    Regras:
    1. Se uma pessoa não atingir 'min_photos' (ou se NENHUMA pessoa atingir):
       - As fotos e vídeos são direcionados para a 'Aba Geral' (ou pasta 'Geral').
       - As mídias são agrupadas pelo rosto mais comum (cluster facial de maior frequência).
    2. Se um arquivo contiver mais de um rosto, ele é associado prioritariamente
       ao rosto mais comum (de maior frequência no conjunto).
    3. Pessoas com 'images_count >= min_photos' permanecem em 'main_people'
       (a menos que nenhuma atinja o mínimo, situação em que todas vão para a aba geral).
    """
    if not people:
        return {
            "all_in_general": True,
            "min_photos": min_photos,
            "main_people": [],
            "general_people": [],
            "no_faces": list(files_without_faces or []),
            "file_to_primary_face": {}
        }

    min_photos = max(1, int(min_photos))

    # Mapeia a frequência de cada pessoa (índice 0 = mais frequente / rosto mais comum)
    file_to_person_indices: Dict[Path, List[int]] = {}
    for p_idx, person in enumerate(people):
        for f in person.get("files", []):
            file_to_person_indices.setdefault(f, []).append(p_idx)

    # Identifica para cada arquivo qual é o rosto mais comum presente nele
    file_to_primary_face: Dict[Path, Dict[str, Any]] = {}
    for f, p_indices in file_to_person_indices.items():
        best_idx = min(p_indices)  # Menor índice = pessoa mais comum/frequente
        file_to_primary_face[f] = people[best_idx]

    # Verifica quem atinge a quantidade mínima de fotos
    qualifying_indices = [
        idx for idx, p in enumerate(people)
        if p.get("images_count", 0) >= min_photos
    ]

    all_in_general = (len(qualifying_indices) == 0)

    if all_in_general:
        # Nenhuma pessoa atingiu o mínimo de fotos:
        # TODAS as mídias vão para a Aba Geral, agrupadas pelo rosto mais comum
        main_people: List[Dict[str, Any]] = []

        person_grouped_files: Dict[int, List[Path]] = {}
        for f, best_p in file_to_primary_face.items():
            best_idx = next(i for i, p in enumerate(people) if p["person_id"] == best_p["person_id"])
            person_grouped_files.setdefault(best_idx, []).append(f)

        general_people: List[Dict[str, Any]] = []
        for p_idx, person in enumerate(people):
            assigned_files = person_grouped_files.get(p_idx, [])
            if not assigned_files:
                continue

            img_count = sum(1 for f in assigned_files if f.suffix.lower() in IMAGE_EXTENSIONS)
            vid_count = len(assigned_files) - img_count

            g_person = dict(person)
            g_person["files"] = assigned_files
            g_person["images_count"] = img_count
            g_person["videos_count"] = vid_count
            g_person["total_media"] = len(assigned_files)
            general_people.append(g_person)

        # Ordena grupos do geral pelo rosto mais comum (quantidade de mídias decrescente)
        general_people.sort(key=lambda p: p["total_media"], reverse=True)
        for i, g in enumerate(general_people, start=1):
            g["rank"] = i
            g["is_most_common"] = (i == 1)

    else:
        # Algumas pessoas atingiram o mínimo de fotos
        main_people = [people[idx] for idx in qualifying_indices]
        non_qualifying_indices = [
            idx for idx in range(len(people))
            if idx not in qualifying_indices
        ]

        # Arquivos das pessoas que não qualificaram vão para a Aba Geral,
        # agrupadas pelo rosto mais comum entre elas.
        general_people = []
        if non_qualifying_indices:
            non_qual_grouped_files: Dict[int, List[Path]] = {}
            for p_idx in non_qualifying_indices:
                person = people[p_idx]
                for f in person.get("files", []):
                    primary_p = file_to_primary_face.get(f)
                    if primary_p and primary_p["person_id"] in [people[i]["person_id"] for i in qualifying_indices]:
                        continue  # Arquivo já está associado a uma pessoa principal
                    
                    eligible_non_qual = [idx for idx in file_to_person_indices.get(f, []) if idx in non_qualifying_indices]
                    if eligible_non_qual:
                        best_non_qual_idx = min(eligible_non_qual)
                        if f not in non_qual_grouped_files.setdefault(best_non_qual_idx, []):
                            non_qual_grouped_files[best_non_qual_idx].append(f)

            for p_idx in non_qualifying_indices:
                assigned_files = non_qual_grouped_files.get(p_idx, [])
                if not assigned_files:
                    continue

                person = people[p_idx]
                img_count = sum(1 for f in assigned_files if f.suffix.lower() in IMAGE_EXTENSIONS)
                vid_count = len(assigned_files) - img_count

                g_person = dict(person)
                g_person["files"] = assigned_files
                g_person["images_count"] = img_count
                g_person["videos_count"] = vid_count
                g_person["total_media"] = len(assigned_files)
                general_people.append(g_person)

            general_people.sort(key=lambda p: p["total_media"], reverse=True)
            for i, g in enumerate(general_people, start=1):
                g["rank"] = i
                g["is_most_common"] = (i == 1)

    return {
        "all_in_general": all_in_general,
        "min_photos": min_photos,
        "main_people": main_people,
        "general_people": general_people,
        "no_faces": list(files_without_faces or []),
        "file_to_primary_face": file_to_primary_face
    }

class FaceClusterer:
    def __init__(self, similarity_threshold: float = 0.38):
        """
        similarity_threshold: Similaridade de cosseno mínima (0.0 a 1.0) para considerar a mesma pessoa.
        Recomendado pelo OpenCV SFace: ~0.363 a 0.40.
        """
        self.similarity_threshold = float(similarity_threshold)

    def group_by_min_photos(
        self,
        people: List[Dict[str, Any]],
        min_photos: int = 3,
        files_without_faces: Optional[List[Path]] = None,
        all_faces: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Método de conveniência que chama group_by_min_photos."""
        return group_by_min_photos(
            people=people,
            min_photos=min_photos,
            files_without_faces=files_without_faces,
            all_faces=all_faces
        )

    def cluster_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Agrupa faces por similaridade e retorna a lista de pessoas identificadas.
        """
        if not faces:
            return []

        if len(faces) == 1:
            face = faces[0]
            return [{
                "person_id": 1,
                "name": "Pessoa_01",
                "avatar_crop": face["crop"],
                "face_indices": [0],
                "files": [face["file_path"]],
                "images_count": 1 if face["media_type"] == "image" else 0,
                "videos_count": 1 if face["media_type"] == "video" else 0,
            }]

        # Matriz de embeddings N x 128
        embeddings = np.array([f["embedding"] for f in faces], dtype=np.float32)
        
        # Similaridade de cosseno pairwise: S_ij = E_i . E_j (pois já estão normalizados)
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Distância de cosseno: D_ij = 1 - S_ij
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

        # Distância máxima de corte correspondente ao threshold de similaridade
        dist_threshold = max(0.01, 1.0 - self.similarity_threshold)

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=dist_threshold,
            n_clusters=None
        )
        labels = clustering.fit_predict(dist_matrix)

        # Agrupa índices de faces por rótulo de cluster
        clusters_map: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            clusters_map.setdefault(lbl, []).append(idx)

        # Constrói objetos de pessoas
        people: List[Dict[str, Any]] = []
        for raw_id, face_indices in clusters_map.items():
            # Encontra a melhor face para avatar do grupo
            # (aquela com maior score e maior similaridade média com os membros do grupo)
            cluster_embeddings = embeddings[face_indices]
            if len(face_indices) > 1:
                # Média de similaridade interna
                sub_sim = np.dot(cluster_embeddings, cluster_embeddings.T)
                avg_sims = np.mean(sub_sim, axis=1)
                best_sub_idx = int(np.argmax(avg_sims))
                best_face_idx = face_indices[best_sub_idx]
            else:
                best_face_idx = face_indices[0]

            avatar_crop = faces[best_face_idx]["crop"]

            # Arquivos únicos onde essa pessoa aparece
            unique_files = list(dict.fromkeys([faces[i]["file_path"] for i in face_indices]))
            img_count = sum(1 for p in unique_files if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'})
            vid_count = len(unique_files) - img_count

            people.append({
                "raw_id": raw_id,
                "avatar_crop": avatar_crop,
                "face_indices": face_indices,
                "files": unique_files,
                "images_count": img_count,
                "videos_count": vid_count,
                "total_media": len(unique_files)
            })

        # Ordena pessoas pela quantidade de mídias (a pessoa mais frequente vira Pessoa_01)
        people.sort(key=lambda p: p["total_media"], reverse=True)

        # Atribui nomes amigáveis ordenados
        for i, person in enumerate(people, start=1):
            person["person_id"] = i
            person["name"] = f"Pessoa_{i:02d}"

        return people
