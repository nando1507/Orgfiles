import sys
import argparse
from pathlib import Path

# Garante suporte a UTF-8 no terminal do Windows sem erros de codificação
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Fallback gracioso caso o tqdm não esteja instalado no Python ativo
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=0, desc="", unit=""):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            pct = int((self.n / max(1, self.total)) * 100)
            print(f"\r[*] {self.desc}: {self.n}/{self.total} ({pct}%)", end="", flush=True)
        def set_postfix_str(self, s):
            pass
        def close(self):
            print()

from core.detector import FaceModels
from core.processor import MediaProcessor
from core.cluster import FaceClusterer, group_by_min_photos
from core.organizer import FileOrganizer

def print_banner():
    banner = """
=============================================================
  ORGANIZADOR INTELIGENTE DE FOTOS E VÍDEOS POR ROSTO
  (Detecção YuNet + Reconhecimento Facial SFace com OpenCV)
=============================================================
"""
    print(banner)

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Organiza fotos e vídeos em pastas por similaridade de rostos."
    )
    parser.add_argument("-i", "--input", help="Caminho da pasta de origem com fotos e vídeos")
    parser.add_argument("-o", "--output", help="Caminho da pasta de destino das pastas organizadas")
    parser.add_argument(
        "-a", "--action",
        choices=["move", "copy"],
        default="move",
        help="Ação a ser executada: 'move' (mover arquivos originais) ou 'copy' (copiar mantendo os originais). Padrão: move"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.38,
        help="Limiar de similaridade de cosseno (0.0 a 1.0). Maior = mais rígido (mesma pessoa). Padrão: 0.38"
    )
    parser.add_argument(
        "-m", "--media-type",
        choices=["both", "images", "videos", "ambos", "imagens", "fotos"],
        default="both",
        help="Tipo de mídia a processar: 'both' (ambos), 'images' (apenas fotos) ou 'videos' (apenas vídeos). Padrão: both"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Taxa de amostragem de quadros por segundo para vídeos. Padrão: 1.0"
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=50.0,
        help="Tamanho máximo de arquivo em MB para processar. Arquivos maiores serão ignorados. Use 0 para desativar. Padrão: 50.0 MB"
    )
    parser.add_argument(
        "--min-photos",
        type=int,
        default=1,
        help="Quantidade mínima de fotos por pessoa. Se não atingir, arquivos vão para a pasta 'Geral', agrupados pelo rosto mais comum. Padrão: 1"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa em modo de simulação (não move nem copia nenhum arquivo)"
    )
    parser.add_argument(
        "--no-separate-unidentified",
        action="store_true",
        help="Não move arquivos sem rostos para a pasta 'Sem_Rosto_Detectado'"
    )

    args = parser.parse_args()

    # Normaliza o tipo de mídia
    mt_raw = args.media_type.lower()
    if mt_raw in ["images", "imagens", "fotos"]:
        media_type = "images"
        media_label = "Apenas Fotos"
    elif mt_raw in ["videos", "vídeos"]:
        media_type = "videos"
        media_label = "Apenas Vídeos"
    else:
        media_type = "both"
        media_label = "Ambos (Fotos e Vídeos)"

    # Modo interativo caso o usuário rode sem argumentos
    input_dir = args.input
    if not input_dir:
        input_dir = input("[*] Digite o caminho da pasta de fotos e vídeos: ").strip().strip('"').strip("'")
        if not input_dir:
            print("[!] Caminho de entrada é obrigatório!")
            sys.exit(1)

        user_mt = input("[*] Processar: [1] Fotos e Vídeos, [2] Apenas Fotos, [3] Apenas Vídeos [Padrão: 1]: ").strip()
        if user_mt == "2":
            media_type = "images"
            media_label = "Apenas Fotos"
        elif user_mt == "3":
            media_type = "videos"
            media_label = "Apenas Vídeos"

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"[!] O diretório informado não existe: {input_path}")
        sys.exit(1)

    output_dir = args.output
    if not output_dir:
        default_out = input_path / "pessoas_organizadas"
        user_out = input(f"[*] Pasta de destino [Padrão: {default_out}]: ").strip().strip('"').strip("'")
        output_path = Path(user_out) if user_out else default_out
    else:
        output_path = Path(output_dir)

    action = args.action
    dry_run = args.dry_run
    max_mb = args.max_size_mb if args.max_size_mb > 0 else None

    print("\n-------------------------------------------------------------")
    print(f" Origem:        {input_path.resolve()}")
    print(f" Destino:       {output_path.resolve()}")
    print(f" Mídia:         {media_label}")
    print(f" Ação:          {'MOVER' if action == 'move' else 'COPIAR'}")
    print(f" Similaridade:  {args.threshold}")
    print(f" Amostra Vídeo: {args.fps} fps")
    print(f" Limite Tamanho:{f'{max_mb} MB (arquivos maiores serão pulados)' if max_mb else 'Sem limite'}")
    print(f" Modo Simulado: {'SIM (nenhum arquivo será movido)' if dry_run else 'NÃO (execução real)'}")
    print("-------------------------------------------------------------\n")

    # 1. Carrega os modelos neurais
    print("[*] Carregando modelos neurais de detecção e reconhecimento facial...")
    models = FaceModels(models_dir="models")

    # 2. Varre os arquivos
    processor = MediaProcessor(
        models=models,
        video_sample_fps=args.fps,
        max_file_size_mb=max_mb,
        media_type=media_type
    )
    media_files = processor.list_media_files(input_path, media_type=media_type)

    if not media_files:
        print(f"[!] Nenhum arquivo de {media_label.lower()} encontrado na pasta de origem.")
        sys.exit(0)

    print(f"\n[*] Encontrados {len(media_files)} arquivos de mídia. Iniciando análise facial...")

    # Barra de progresso para a varredura
    scan_pbar = tqdm(total=len(media_files), desc="Analisando mídias", unit="arquivo")
    def on_scan_progress(current, total, filename):
        scan_pbar.set_postfix_str(filename[:25])
        scan_pbar.update(1)

    scan_result = processor.scan_directory(media_files, progress_callback=on_scan_progress)
    scan_pbar.close()

    all_faces = scan_result["all_faces"]
    no_faces = scan_result["files_without_faces"]
    skipped_large = scan_result["skipped_large_files"]

    processed_count = len(media_files) - len(no_faces) - len(skipped_large)
    print(f"\n[+] Análise concluída: {len(all_faces)} faces detectadas em {processed_count} arquivos.")
    if skipped_large:
        print(f"[!] {len(skipped_large)} arquivos foram pulados por excederem {max_mb} MB.")
    if no_faces:
        print(f"[-] {len(no_faces)} arquivos analisados não tiveram rostos detectados.")

    # 3. Agrupamento (Clustering)
    print("\n[*] Agrupando rostos semelhantes...")
    clusterer = FaceClusterer(similarity_threshold=args.threshold)
    people = clusterer.cluster_faces(all_faces)

    # Separa por quantidade mínima de fotos e agrupa pelo rosto mais comum
    group_results = group_by_min_photos(
        people=people,
        min_photos=args.min_photos,
        files_without_faces=no_faces,
        all_faces=all_faces
    )
    all_in_general = group_results["all_in_general"]
    main_people = group_results["main_people"]
    general_people = group_results["general_people"]

    if all_in_general:
        print(f"\n[!] Nenhuma pessoa atingiu a quantidade mínima de {args.min_photos} fotos.")
        print(f"[*] Todas as mídias foram direcionadas para a pasta 'Geral', agrupadas pelo rosto mais comum:")
        for g in general_people:
            print(f"   -> {g['name']}: {g['total_media']} mídias ({g['images_count']} fotos, {g['videos_count']} vídeos)")
    else:
        print(f"\n[+] Pessoas Principais (>= {args.min_photos} fotos): {len(main_people)}")
        for p in main_people:
            print(f"   -> {p['name']}: {p['total_media']} mídias ({p['images_count']} fotos, {p['videos_count']} vídeos)")
        if general_people:
            print(f"[+] Mídias na pasta Geral (< {args.min_photos} fotos, agrupadas pelo rosto mais comum): {len(general_people)} grupos")
            for g in general_people:
                print(f"   -> {g['name']}: {g['total_media']} mídias ({g['images_count']} fotos, {g['videos_count']} vídeos)")

    # 4. Organização / Movimentação dos arquivos
    print("\n[*] Iniciando organização das pastas...")
    organizer = FileOrganizer(
        destination_dir=output_path,
        action=action,
        organize_no_faces=not args.no_separate_unidentified,
        save_avatars=True,
        dry_run=dry_run
    )

    org_pbar = tqdm(total=len(media_files) - len(skipped_large), desc="Organizando arquivos", unit="arquivo")
    def on_org_progress(current, total, filename):
        org_pbar.set_postfix_str(filename[:25])
        org_pbar.update(1)

    stats = organizer.execute(
        people=main_people,
        files_without_faces=no_faces,
        progress_callback=on_org_progress,
        general_people=general_people,
        all_in_general=all_in_general
    )
    org_pbar.close()

    print("\n=============================================================")
    print("                    RELATÓRIO FINAL                          ")
    print("=============================================================")
    print(f"• Pessoas principais:            {stats['people_count']}")
    print(f"• Mídias no Geral:               {stats.get('general_count', 0)}")
    print(f"• Arquivos movidos:              {stats['moved_count']}")
    print(f"• Arquivos copiados:             {stats['copied_count']}")
    print(f"• Arquivos sem rosto:            {stats['no_faces_count']}")
    print(f"• Arquivos pulados (> {max_mb or 0} MB):   {len(skipped_large)}")
    if stats["errors"]:
        print(f"• Erros encontrados:             {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"  - {err}")
    print(f"• Pasta final de destino:        {output_path.resolve()}")
    print("=============================================================\n")

if __name__ == "__main__":
    main()
