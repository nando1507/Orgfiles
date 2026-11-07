import os
import shutil
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

def get_unique_destination_path(target_folder: Path, filename: str) -> Path:
    """Evita sobrescrita de arquivos adicionando um sufixo numérico caso já exista um com o mesmo nome."""
    dest = target_folder / filename
    if not dest.exists():
        return dest
    
    stem = dest.stem
    suffix = dest.suffix
    counter = 1
    while True:
        candidate = target_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

class FileOrganizer:
    def __init__(
        self,
        destination_dir: Path,
        action: str = "move",  # "move" ou "copy"
        organize_no_faces: bool = True,
        save_avatars: bool = True,
        dry_run: bool = False
    ):
        self.destination_dir = Path(destination_dir)
        self.action = action.lower()
        self.organize_no_faces = organize_no_faces
        self.save_avatars = save_avatars
        self.dry_run = dry_run

    def execute(
        self,
        people: List[Dict[str, Any]],
        files_without_faces: List[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        general_people: Optional[List[Dict[str, Any]]] = None,
        all_in_general: bool = False,
        general_folder_name: str = "Geral"
    ) -> Dict[str, Any]:
        """
        Executa a organização dos arquivos conforme os grupos de pessoas identificadas.
        Se all_in_general for True ou se houver general_people, organiza as mídias da aba geral
        na pasta 'Geral' agrupadas pelo respectivo rosto mais comum.
        
        Se um arquivo tiver múltiplas pessoas:
        - No modo 'move': move para a pasta da primeira pessoa e copia para as pastas das demais.
        - No modo 'copy': copia para todas as pastas das pessoas detectadas.
        """
        if not self.dry_run:
            self.destination_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_files_processed": 0,
            "moved_count": 0,
            "copied_count": 0,
            "no_faces_count": 0,
            "general_count": 0,
            "people_count": len(people) if not all_in_general else 0,
            "general_groups_count": len(general_people or []) if not all_in_general else len(people),
            "errors": []
        }

        file_to_targets: Dict[Path, List[Path]] = {}
        general_folders_set = set()

        if all_in_general:
            # Todas as mídias vão para a pasta Geral, agrupadas pelo rosto mais comum
            general_base = self.destination_dir / general_folder_name
            groups = general_people if general_people is not None else people
            for p in groups:
                p_id = p["person_id"]
                p_name = p.get("name", f"Pessoa_{p_id:02d}")
                folder_path = general_base / p_name
                general_folders_set.add(folder_path)

                if not self.dry_run:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    if self.save_avatars and p.get("avatar_crop") is not None:
                        avatar_path = folder_path / "_foto_referencia.jpg"
                        if not avatar_path.exists():
                            try:
                                cv2.imencode(".jpg", p["avatar_crop"])[1].tofile(str(avatar_path))
                            except Exception as e:
                                stats["errors"].append(f"Erro ao salvar avatar de {p_name}: {e}")

                for file_path in p["files"]:
                    file_to_targets.setdefault(file_path, []).append(folder_path)

            no_face_folder = general_base / "Sem_Rosto_Detectado"
            if self.organize_no_faces and files_without_faces:
                if not self.dry_run:
                    no_face_folder.mkdir(parents=True, exist_ok=True)
                for file_path in files_without_faces:
                    file_to_targets.setdefault(file_path, []).append(no_face_folder)

        else:
            # Pessoas principais recebem pastas individuais na raiz do destino
            person_folders: Dict[int, Path] = {}
            for p in people:
                p_id = p["person_id"]
                p_name = p.get("name", f"Pessoa_{p_id:02d}")
                folder_path = self.destination_dir / p_name
                person_folders[p_id] = folder_path

                if not self.dry_run:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    if self.save_avatars and p.get("avatar_crop") is not None:
                        avatar_path = folder_path / "_foto_referencia.jpg"
                        if not avatar_path.exists():
                            try:
                                cv2.imencode(".jpg", p["avatar_crop"])[1].tofile(str(avatar_path))
                            except Exception as e:
                                stats["errors"].append(f"Erro ao salvar avatar de {p_name}: {e}")

                for file_path in p["files"]:
                    file_to_targets.setdefault(file_path, []).append(folder_path)

            # Mídias da aba geral vão para subpastas de Geral/ agrupadas pelo rosto
            if general_people:
                general_base = self.destination_dir / general_folder_name
                for g in general_people:
                    g_id = g["person_id"]
                    g_name = g.get("name", f"Pessoa_{g_id:02d}")
                    g_folder = general_base / g_name
                    general_folders_set.add(g_folder)

                    if not self.dry_run:
                        g_folder.mkdir(parents=True, exist_ok=True)
                        if self.save_avatars and g.get("avatar_crop") is not None:
                            avatar_path = g_folder / "_foto_referencia.jpg"
                            if not avatar_path.exists():
                                try:
                                    cv2.imencode(".jpg", g["avatar_crop"])[1].tofile(str(avatar_path))
                                except Exception as e:
                                    stats["errors"].append(f"Erro ao salvar avatar de {g_name}: {e}")

                    for file_path in g["files"]:
                        file_to_targets.setdefault(file_path, []).append(g_folder)

            no_face_folder = self.destination_dir / "Sem_Rosto_Detectado"
            if self.organize_no_faces and files_without_faces:
                if not self.dry_run:
                    no_face_folder.mkdir(parents=True, exist_ok=True)
                for file_path in files_without_faces:
                    file_to_targets.setdefault(file_path, []).append(no_face_folder)

        total_files = len(file_to_targets)
        for i, (src_file, targets) in enumerate(file_to_targets.items(), start=1):
            if progress_callback:
                progress_callback(i, total_files, src_file.name)

            stats["total_files_processed"] += 1

            if not src_file.exists():
                stats["errors"].append(f"Arquivo de origem não encontrado: {src_file}")
                continue

            try:
                # Se for para o diretório de sem rosto
                if targets == [no_face_folder]:
                    stats["no_faces_count"] += 1
                elif any(t in general_folders_set for t in targets):
                    stats["general_count"] += 1

                if self.action == "copy" or self.dry_run:
                    # Copia para todos os destinos
                    for tgt_dir in targets:
                        dest_file = get_unique_destination_path(tgt_dir, src_file.name)
                        if not self.dry_run:
                            shutil.copy2(str(src_file), str(dest_file))
                        stats["copied_count"] += 1
                else:
                    # Modo 'move'
                    # Se houver mais de uma pessoa: copia para todas a partir da 2ª, e move para a 1ª
                    if len(targets) > 1:
                        # Copia para as pastas secundárias
                        for tgt_dir in targets[1:]:
                            dest_file = get_unique_destination_path(tgt_dir, src_file.name)
                            shutil.copy2(str(src_file), str(dest_file))
                            stats["copied_count"] += 1
                        
                        # Move o original para a primeira pasta
                        dest_file = get_unique_destination_path(targets[0], src_file.name)
                        shutil.move(str(src_file), str(dest_file))
                        stats["moved_count"] += 1
                    else:
                        # Uma única pessoa: move diretamente
                        dest_file = get_unique_destination_path(targets[0], src_file.name)
                        shutil.move(str(src_file), str(dest_file))
                        stats["moved_count"] += 1

            except Exception as e:
                stats["errors"].append(f"Erro processando {src_file.name}: {e}")

        return stats
