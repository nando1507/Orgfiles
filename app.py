import os
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from core.detector import FaceModels
from core.processor import MediaProcessor
from core.cluster import FaceClusterer, group_by_min_photos
from core.organizer import FileOrganizer

# Configuração da página Streamlit
st.set_page_config(
    page_title="Organizador Facial de Fotos & Vídeos",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .person-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👤 Organizador Facial Inteligente</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Agrupe fotos e vídeos automaticamente pela semelhança dos rostos detectados.</div>', unsafe_allow_html=True)

# Barra Lateral: Configurações
st.sidebar.header("⚙️ Configurações")

input_folder_str = st.sidebar.text_input(
    "📁 Pasta de Origem (Fotos/Vídeos)",
    value="",
    placeholder="Ex: C:\\MinhasFotos",
    help="Cole o caminho completo da pasta que contém as fotos e vídeos que deseja organizar."
)

output_folder_str = st.sidebar.text_input(
    "📂 Pasta de Destino (Organizados)",
    value="",
    placeholder="Deixe em branco para salvar na subpasta 'organizados'",
    help="Onde as pastas de cada pessoa identificada serão criadas."
)

action_mode = st.sidebar.radio(
    "Ação nos arquivos originais",
    options=["Mover (tirar da pasta original)", "Copiar (manter originais)"],
    index=0,
    help="Mover irá transferir os arquivos para as pastas dos rostos correspondentes. Se uma foto tiver mais de uma pessoa, ela é copiada para as pastas dos outros participantes."
)
action_type = "move" if "Mover" in action_mode else "copy"

media_filter_choice = st.sidebar.selectbox(
    "📁 Tipo de Mídia a Processar",
    options=["Ambos (Fotos e Vídeos)", "Apenas Fotos", "Apenas Vídeos"],
    index=0,
    help="Escolha se deseja processar fotos, vídeos ou ambos."
)
if media_filter_choice == "Apenas Fotos":
    media_filter_type = "images"
elif media_filter_choice == "Apenas Vídeos":
    media_filter_type = "videos"
else:
    media_filter_type = "both"

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Sensibilidade da IA")

similarity_thresh = st.sidebar.slider(
    "Limiar de Similaridade Facial",
    min_value=0.25,
    max_value=0.60,
    value=0.38,
    step=0.01,
    help="Valores mais altos exigem maior certeza para juntar rostos (evita falsos positivos). Valores menores agrupam mais variações (ângulos, óculos, idade). Padrão recomendado: 0.38."
)

video_fps = st.sidebar.slider(
    "Amostragem em Vídeos (quadros/seg)",
    min_value=0.2,
    max_value=3.0,
    value=1.0,
    step=0.2,
    help="Quantos quadros analisar por segundo de vídeo. 1.0 fps é rápido e detecta rostos com ótima precisão."
)

organize_no_faces = st.sidebar.checkbox(
    "Separar arquivos sem rostos detectados",
    value=True,
    help="Se marcado, fotos e vídeos sem rostos visíveis serão movidos/copiados para a pasta 'Sem_Rosto_Detectado'."
)

min_photos = st.sidebar.number_input(
    "Quantidade mínima de fotos por pessoa",
    min_value=1,
    max_value=100,
    value=3,
    step=1,
    help="Se uma pessoa não atingir este número mínimo de fotos (ou se nenhuma pessoa atingir), as mídias irão para a 'Aba Geral', agrupadas pelo rosto mais comum."
)

max_size_mb = st.sidebar.number_input(
    "Tamanho máximo de arquivo (MB)",
    min_value=0,
    max_value=4096,
    value=50,
    step=10,
    help="Pula arquivos maiores que este limite (ex: vídeos muito pesados). Use 0 para desativar o limite."
)

# Inicializa estados de sessão
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "people" not in st.session_state:
    st.session_state.people = []
if "all_faces" not in st.session_state:
    st.session_state.all_faces = []
if "no_faces" not in st.session_state:
    st.session_state.no_faces = []
if "skipped_large" not in st.session_state:
    st.session_state.skipped_large = []
if "media_files" not in st.session_state:
    st.session_state.media_files = []

# Botão principal de análise
col_btn1, col_btn2 = st.columns([2, 5])

input_path = Path(input_folder_str.strip().strip('"').strip("'")) if input_folder_str else None
valid_input = input_path is not None and input_path.exists() and input_path.is_dir()

with col_btn1:
    start_analysis = st.button("🔍 1. Analisar e Agrupar Rostos", type="primary", use_container_width=True, disabled=not valid_input)

if not valid_input and input_folder_str:
    st.error("⚠️ O caminho informado para a pasta de origem não existe ou não é um diretório válido.")

# Execução da Análise
if start_analysis and valid_input:
    # Determina pasta de saída
    if output_folder_str.strip():
        output_path = Path(output_folder_str.strip().strip('"').strip("'"))
    else:
        output_path = input_path / "pessoas_organizadas"
    st.session_state.output_path = output_path

    # Carrega modelos
    with st.spinner("🤖 Carregando inteligência artificial (YuNet + SFace)..."):
        models = FaceModels(models_dir="models")
        limit_mb = float(max_size_mb) if max_size_mb > 0 else None
        processor = MediaProcessor(
            models=models,
            video_sample_fps=video_fps,
            max_file_size_mb=limit_mb,
            media_type=media_filter_type
        )

    # Lista arquivos
    files = processor.list_media_files(input_path, media_type=media_filter_type)
    if not files:
        st.warning(f"Nenhum arquivo ({media_filter_choice.lower()}) encontrado na pasta de origem.")
    else:
        st.session_state.media_files = files
        progress_bar = st.progress(0, text="Iniciando leitura dos arquivos...")
        status_text = st.empty()

        def update_progress(current, total, filename):
            pct = int((current / total) * 100)
            progress_bar.progress(pct, text=f"Processando ({current}/{total}): {filename}")

        scan_result = processor.scan_directory(files, progress_callback=update_progress)
        progress_bar.empty()
        status_text.empty()

        all_faces = scan_result["all_faces"]
        st.session_state.all_faces = all_faces
        st.session_state.no_faces = scan_result["files_without_faces"]
        st.session_state.skipped_large = scan_result["skipped_large_files"]

        # Agrupamento
        with st.spinner("🧠 Agrupando pessoas por semelhança facial..."):
            clusterer = FaceClusterer(similarity_threshold=similarity_thresh)
            people = clusterer.cluster_faces(all_faces)
            st.session_state.people = people
            st.session_state.analysis_done = True

# Apresentação dos Resultados da Análise
if st.session_state.analysis_done:
    people = st.session_state.people
    no_faces = st.session_state.no_faces
    skipped_large = st.session_state.skipped_large
    total_files = len(st.session_state.media_files)
    all_faces = st.session_state.get("all_faces", [])

    # Agrupa por quantidade mínima de fotos e pelo rosto mais comum
    group_results = group_by_min_photos(
        people=people,
        min_photos=min_photos,
        files_without_faces=no_faces,
        all_faces=all_faces
    )

    all_in_general = group_results["all_in_general"]
    main_people = group_results["main_people"]
    general_people = group_results["general_people"]

    if all_in_general:
        st.warning(
            f"ℹ️ Nenhuma pessoa atingiu a quantidade mínima de **{min_photos} fotos**. "
            f"Todas as fotos e vídeos foram reunidos na **Aba Geral**, agrupados pelo **rosto mais comum**."
        )
    else:
        st.success(
            f"✅ Análise concluída! Identificadas **{len(people)} pessoas diferentes** em {total_files} mídias. "
            f"**{len(main_people)} pessoas** atingiram o mínimo de fotos e as demais mídias foram direcionadas para a **Aba Geral**, agrupadas pelo rosto mais comum."
        )

    # Resumo em métricas
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total de Mídias", total_files)
    m2.metric("Pessoas Principais", len(main_people))
    m3.metric("Grupos no Geral", len(general_people))
    m4.metric("Mídias sem Rosto", len(no_faces))
    m5.metric("Pulados (>Limite)", len(skipped_large))

    if skipped_large:
        with st.expander(f"⚠️ Ver arquivos pulados por excederem o limite ({len(skipped_large)})"):
            for f in skipped_large:
                sz = f.stat().st_size / (1024 * 1024) if f.exists() else 0
                st.caption(f"• {f.name} ({sz:.1f} MB)")

    st.markdown("---")
    st.subheader("👥 Visualização e Agrupamento dos Arquivos")

    if all_in_general:
        tab_geral, = st.tabs([f"📁 Aba Geral ({total_files - len(skipped_large)} mídias agrupadas por rosto)"])
        tab_principais = None
    else:
        total_geral_media = sum(g["total_media"] for g in general_people) + len(no_faces)
        tab_principais, tab_geral = st.tabs([
            f"👤 Pessoas Principais ({len(main_people)})",
            f"📁 Aba Geral ({total_geral_media} mídias)"
        ])

    if tab_principais is not None:
        with tab_principais:
            st.info("💡 Pessoas que atingiram a quantidade mínima de fotos. Você pode renomear cada uma antes de organizar os arquivos.")
            cols_per_row = 3
            for i in range(0, len(main_people), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(main_people):
                        p = main_people[idx]
                        with cols[j]:
                            with st.container():
                                st.markdown(f"#### 👤 {p['name']}")
                                if p.get("avatar_crop") is not None:
                                    avatar_rgb = cv2.cvtColor(p["avatar_crop"], cv2.COLOR_BGR2RGB)
                                    st.image(avatar_rgb, caption=f"Rosto de referência ({p['name']})", width=160)

                                st.write(f"📁 **{p['total_media']} arquivos** ({p['images_count']} fotos, {p['videos_count']} vídeos)")

                                new_name = st.text_input(
                                    f"Nome da pasta para {p['name']}:",
                                    value=p['name'],
                                    key=f"rename_main_{p['person_id']}"
                                )
                                p["name"] = new_name.strip() if new_name.strip() else p["name"]

                                with st.expander("Ver arquivos associados"):
                                    for f in p["files"]:
                                        st.caption(f"• {f.name}")

    with tab_geral:
        if all_in_general:
            st.info("💡 Todas as fotos e vídeos foram agrupadas pelo **rosto mais comum** (ordenadas por frequência).")
        else:
            st.info("💡 Mídias que não atingiram a quantidade mínima de fotos, agrupadas pelo **rosto mais comum**.")

        if general_people:
            cols_per_row = 3
            for i in range(0, len(general_people), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(general_people):
                        g = general_people[idx]
                        with cols[j]:
                            with st.container():
                                rank_label = "👑 Rosto Mais Comum #1" if g.get("is_most_common") else f"👤 Rosto Comum #{g.get('rank', idx+1)}"
                                st.markdown(f"#### {rank_label}")
                                if g.get("avatar_crop") is not None:
                                    avatar_rgb = cv2.cvtColor(g["avatar_crop"], cv2.COLOR_BGR2RGB)
                                    st.image(avatar_rgb, caption=f"Rosto ({g['name']})", width=160)

                                st.write(f"📁 **{g['total_media']} arquivos** ({g['images_count']} fotos, {g['videos_count']} vídeos)")

                                new_g_name = st.text_input(
                                    f"Nome da pasta no Geral ({g['name']}):",
                                    value=g['name'],
                                    key=f"rename_general_{g['person_id']}"
                                )
                                g["name"] = new_g_name.strip() if new_g_name.strip() else g["name"]

                                with st.expander(f"Ver arquivos agrupados ({len(g['files'])})"):
                                    for f in g["files"]:
                                        st.caption(f"• {f.name}")
        else:
            st.caption("Nenhuma mídia não agrupada.")

        if no_faces:
            st.markdown("---")
            with st.expander(f"⚠️ Ver arquivos sem rostos detectados ({len(no_faces)})"):
                for f in no_faces:
                    st.caption(f"• {f.name}")

    st.markdown("---")
    st.subheader("🚀 2. Executar Organização")

    dest_folder = st.session_state.get("output_path", input_path / "pessoas_organizadas")
    st.write(f"📍 Os arquivos serão organizados em: `{dest_folder.resolve()}`")
    st.write(f"📌 Ação selecionada: **{'MOVER (os arquivos sairão da pasta de origem)' if action_type == 'move' else 'COPIAR (mantém os originais)'}**")

    col_exec1, col_exec2 = st.columns([2, 5])
    with col_exec1:
        run_organize = st.button("📦 Confirmar e Organizar Arquivos", type="primary", use_container_width=True)

    if run_organize:
        organizer = FileOrganizer(
            destination_dir=dest_folder,
            action=action_type,
            organize_no_faces=organize_no_faces,
            save_avatars=True,
            dry_run=False
        )

        org_pbar = st.progress(0, text="Movendo arquivos...")

        def update_org(curr, tot, fname):
            pct = int((curr / tot) * 100)
            org_pbar.progress(pct, text=f"Organizando ({curr}/{tot}): {fname}")

        with st.spinner("Transferindo e organizando pastas..."):
            stats = organizer.execute(
                people=main_people,
                files_without_faces=no_faces,
                progress_callback=update_org,
                general_people=general_people,
                all_in_general=all_in_general
            )

        org_pbar.empty()
        st.balloons()
        st.success("🎉 Organização concluída com sucesso!")

        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Arquivos Movidos", stats["moved_count"])
        res2.metric("Arquivos Copiados", stats["copied_count"])
        res3.metric("Mídias no Geral", stats.get("general_count", 0))
        res4.metric("Sem Rosto", stats["no_faces_count"])

        if stats["errors"]:
            st.error(f"Ocorreram {len(stats['errors'])} erros:")
            for err in stats["errors"]:
                st.write(f"• {err}")
        else:
            st.info(f"✨ As pastas foram criadas em: `{dest_folder.resolve()}`")
