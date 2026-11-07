# 👤 Organizador Facial Inteligente de Fotos e Vídeos

Sistema em Python que analisa a semelhança entre rostos presentes em **fotos** e **vídeos** utilizando redes neurais profundas (Deep Learning) e agrupa/move os arquivos automaticamente para pastas separadas por pessoa.

---

## 🚀 Tecnologias e Modelos Utilizados

- **Detecção Facial:** `YuNet` (OpenCV Zoo) — detector ultrarrápido e preciso com suporte a 5 pontos de landmarks faciais.
- **Reconhecimento Facial (Embeddings):** `SFace` — rede neural profunda que gera vetores de características de 128 dimensões alinhados com a face.
- **Processamento de Mídia:** OpenCV (`cv2`) para fotos e amostragem inteligente de quadros em vídeos.
- **Agrupamento (Clustering):** `Scikit-Learn` (Clustering Hierárquico Aglomerativo com distância por cosseno).
- **Interface Gráfica:** `Streamlit` para visualização dos rostos em tempo real e renomeação de pastas.
- **Linha de Comando:** `CLI` interativa e automatizada com barras de progresso via `tqdm`.

> 💡 **Vantagem:** Não necessita compilar `dlib` nem instalar ferramentas pesadas de C++. Funciona nativamente no Windows com Python moderno (3.10 a 3.13).

---

## 📂 Estrutura do Projeto

```text
OrgFiles/
├── app.py               # Interface Gráfica Web interativa (Streamlit)
├── main.py              # Script executável via Terminal / CLI
├── requirements.txt     # Dependências do projeto
├── models/              # Modelos neurais ONNX (baixados automaticamente)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
└── core/                # Módulos do sistema
    ├── detector.py      # Carregamento do YuNet e SFace
    ├── processor.py     # Leitura e extração de faces em fotos e vídeos
    ├── cluster.py       # Agrupamento e cálculo de similaridade
    └── organizer.py     # Mecanismo de movimentação, cópia e salvamento de avatares
```

---

## 💻 Como Usar

### Opção 1: Interface Visual Web (Recomendado)

Inicie a aplicação visual com o comando:

```bash
streamlit run app.py
```

Isso abrirá automaticamente seu navegador com uma interface onde você pode:
1. Informar a pasta de origem das fotos e vídeos.
2. Clicar em **"1. Analisar e Agrupar Rostos"**.
3. Ver na tela os **recortes dos rostos** de cada pessoa identificada (`Pessoa_01`, `Pessoa_02`, etc.).
4. **Renomear diretamente na tela** para o nome real da pessoa (ex: "João", "Maria").
5. Clicar em **"2. Confirmar e Organizar Arquivos"** para mover ou copiar tudo com 1 clique!

---

### Opção 2: Linha de Comando (CLI / Terminal)

#### Modo Interativo Simples:
Basta executar:
```bash
python main.py
```
O script perguntará a pasta de origem e destino diretamente no terminal.

#### Modo com Parâmetros / Automatizado:
```bash
# Mover arquivos originais organizando por pessoa (padrão):
python main.py -i "C:\MinhasFotos" -o "C:\FotosOrganizadas" --action move

# Apenas copiar mantendo os originais intactos:
python main.py -i "C:\MinhasFotos" -o "C:\FotosOrganizadas" --action copy

# Modo simulação (testa a análise sem alterar nenhum arquivo):
python main.py -i "C:\MinhasFotos" -o "C:\FotosOrganizadas" --dry-run
```

---

## ⚙️ Parâmetros Disponíveis

| Parâmetro | Descrição | Padrão |
| :--- | :--- | :--- |
| `-i`, `--input` | Pasta de fotos e vídeos a analisar | *(Interativo)* |
| `-o`, `--output` | Pasta onde as pastas das pessoas serão criadas | `<input>/pessoas_organizadas` |
| `-m`, `--media-type` | Filtrar mídias: `both` (ambos), `images` (apenas fotos) ou `videos` (apenas vídeos) | `both` |
| `-a`, `--action` | `move` (move tirando da origem) ou `copy` (copia) | `move` |
| `-t`, `--threshold` | Limiar de similaridade de cosseno (0.0 a 1.0). Quanto maior, mais rígido | `0.38` |
| `--fps` | Quadros por segundo analisados em vídeos | `1.0` |
| `--max-size-mb` | Tamanho máximo em MB para processar (pula arquivos maiores). 0 para desativar | `50.0` |
| `--min-photos` | Mínimo de fotos por pessoa. Se não atingir, vai para a aba/pasta 'Geral' agrupado pelo rosto mais comum | `1` (CLI) / `3` (Web) |
| `--dry-run` | Executa a análise sem mover/copiar arquivos | `False` |

---

## 👥 Como Funciona com Múltiplas Pessoas, Vídeos e Aba Geral?

1. **Fotos com mais de uma pessoa:** O arquivo é associado prioritariamente ao rosto mais comum (de maior frequência). No modo de organização individual por pessoa, o original é movido para o rosto mais frequente e copiado para as demais pessoas presentes.
2. **Aba Geral e Quantidade Mínima de Fotos:** Caso uma pessoa não atinja a quantidade mínima de fotos configurada (ou caso nenhuma pessoa atinja), todas as mídias são reunidas na **Aba Geral** (ou pasta `Geral/`), sendo agrupadas pelo **rosto mais comum** em ordem decrescente de frequência.
3. **Vídeos:** O sistema amostra quadros do vídeo (ex: 1 quadro por segundo), identifica os rostos presentes e associa o arquivo de vídeo à pasta das pessoas que aparecem nele.
4. **Fotos sem rostos visíveis:** São separadas na pasta `Sem_Rosto_Detectado`.
5. **Foto de Referência:** Dentro de cada pasta é salvo um arquivo `_foto_referencia.jpg` com o recorte facial de referência para identificação imediata.
