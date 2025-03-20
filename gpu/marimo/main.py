import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# GPU-accelerated Bioinfromatics Data Processing""")
    return


@app.cell
def _():
    # Core dependencies for data processing and visualization
    from pprint import pprint
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Optional
    import subprocess

    import marimo as mo
    import polars as pl
    import biobear as bb
    from tqdm import tqdm


    # Visualization stack
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D


    # GPU-accelerated computing
    import numpy as np
    import cupy as cp
    from cuml.cluster import KMeans
    from cuml.manifold import UMAP


    # Protein language models
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    import torch


    # AI services integration
    from openai import OpenAI
    return (
        Axes3D,
        ESMC,
        ESMProtein,
        KMeans,
        LogitsConfig,
        OpenAI,
        Optional,
        Path,
        UMAP,
        bb,
        cp,
        dataclass,
        mcolors,
        mo,
        np,
        pl,
        plt,
        pprint,
        subprocess,
        torch,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Fetch Table With Immune-Related Proteins in Archaea

        ### UniProt Query Strategy
        We retrieve archaeal defense proteins from UniProt using a search query that combines taxonomy restriction (Archaea: taxid 2157) with functional keywords related to immunity and defense mechanisms.

        We use such a search command: `((taxonomy_id:2157) AND ((cc_function:immun*) OR (cc_function:defens*) OR (cc_function:antimicrob*) OR (cc_function:antibacter*)))`
        """
    )
    return


@app.cell
def _(pl):
    uniprot_arch_defense_proteins_df = pl.read_csv(
        "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Cxref_string%2Cxref_interpro%2Corganism_id%2Csequence%2Cxref_pdb%2Ccc_function&format=tsv&query=%28%28taxonomy_id%3A2157%29+AND+%28%28cc_function%3Aimmun*%29+OR+%28cc_function%3Adefens*%29+OR+%28cc_function%3Aantimicrob*%29+OR+%28cc_function%3Aantibacter*%29%29%29",
        separator='\t'
    )
    return (uniprot_arch_defense_proteins_df,)


@app.cell
def _(uniprot_arch_defense_proteins_df):
    uniprot_arch_defense_proteins_df.head()
    return


@app.cell
def _(uniprot_arch_defense_proteins_df):
    uniprot_arch_defense_proteins_df.shape
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## CRISPR System Exclusion
        Filter out CRISPR-associated proteins to focus on non-CRISPR defense mechanisms.
        We exclude entries containing "crispr" in three key fields:

        1. Function annotations
        2. Protein names
        3. Gene names
        """
    )
    return


@app.cell
def _(pl, uniprot_arch_defense_proteins_df):
    uniprot_arch_defense_proteins_df_non_crispr = uniprot_arch_defense_proteins_df.filter(
        ~pl.col('Function [CC]').str.to_lowercase().str.contains('crispr') &
        ~pl.col('Protein names').str.to_lowercase().str.contains('crispr') &
        ~pl.col('Gene Names').str.to_lowercase().str.contains('crispr')
    )

    uniprot_arch_defense_proteins_df_non_crispr.head()
    return (uniprot_arch_defense_proteins_df_non_crispr,)


@app.cell
def _(uniprot_arch_defense_proteins_df_non_crispr):
    uniprot_arch_defense_proteins_df_non_crispr.shape
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## MMseqs2 Homology Search Setup
        Prepare directory structure for MMseqs2 workflow:

        - `queryDB`: Contains our curated non-CRISPR defense proteins
        - `targetDB`: Stores archaeal sequences from NCBI nr database
        - `search_results`: Output directory for homology matches
        - `tmp`: Temporary working directory for MMseqs2

        This organized structure ensures reproducibility and simplifies intermediate file management.
        """
    )
    return


@app.cell
def _(run_command):
    run_command(
        [
            "mkdir", "-p" ,
            "./mmseqs_db/queryDB",
            "./mmseqs_db/targetDB",
            "./mmseqs_results/search_results",
            "./mmseqs_results/tmp",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Sequence Data Preparation
        Convert filtered DataFrame to FASTA format for MMseqs2 processing
        """
    )
    return


@app.cell
def _(pl, save_to_fasta, uniprot_arch_defense_proteins_df_non_crispr):
    archaeal_defense_sequences = uniprot_arch_defense_proteins_df_non_crispr. \
        select([pl.col('Entry'), pl.col('Sequence')]). \
        to_dicts()
    save_to_fasta(archaeal_defense_sequences, './mmseqs_db/queryDB/archaeal_defense_sequences.fasta')
    return (archaeal_defense_sequences,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### MMseqs2 Database Creation
        Convert FASTA files to MMseqs2's optimized database format using `createdb` command
        """
    )
    return


@app.cell
def _(run_command):
    run_command(
        [
            "mmseqs", "createdb",
            "./mmseqs_db/queryDB/archaeal_defense_sequences.fasta", "./mmseqs_db/queryDB/queryDB"
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### NCBI nr Database Acquisition
        We can download the entire `nr` database from Nebius Blob Storage (`s3://lshc/datasets/ncbi/blast/db/nr/`), decompress it and extraxct archaeal sequences manually, but we prepared archaeal subset already:
        """
    )
    return


@app.cell
def _(run_command):
    run_command(
        [
            "aws", "s3", "cp", "s3://lshc/examples/archaeal-webinar/archaea.fa", "./mmseqs_db/targetDB/archaea.fa"
        ]
    )
    return


@app.cell
def _(run_command):
    run_command(
        [
            "mmseqs", "createdb",
            "./mmseqs_db/targetDB/archaea.fa", "./mmseqs_db/targetDB/targetDB"
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### GPU Acceleration Setup
        MMseqs2 GPU optimization steps:

        1. **makepaddedseqdb**: Prepares sequences for GPU processing
        2. **createindex**: Builds GPU-optimized search index
        3. **search**: Executes alignment on NVIDIA GPUs
        """
    )
    return


@app.cell
def _(run_command):
    # Prepare GPU-compatible database format
    run_command(
        [
            "mmseqs", "makepaddedseqdb",
            "./mmseqs_db/targetDB/targetDB",
            "./mmseqs_db/targetDB/targetDB_gpu"
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### MMseqs2 Search Execution
        Run iterative profile search with these parameters:
        ```shell
        mmseqs search queryDB targetDB resultDB tmp \

          --comp-bias-corr 1 \              # Composition bias correction
          -s 7.5 \                          # Max sensitivity preset
          --max-seqs 100 \                 # Allow more hits through prefilter
          --spaced-kmer-mode 1 \            # Better distant homology detection
          --exact-kmer-matching 0 \         # Allow approximate k-mer matches
          -a 1 \                            # Generate alignment data
          --alignment-mode 3 \              # Include sequence identity in output
          --num-iterations 3 \              # PSI-BLAST-like iterative profile search
          -e 1e-3 \                         # Keep default E-value threshold
          -c 0.2 --cov-mode 5 \             # 20% coverage of shorter sequence
          --realign 1 \                     # More accurate alignments
          --mask 0 \                        # Disable low-complexity filtering
          --min-aln-len 30 \                # Minimum meaningful alignment length
          --gpu 1 \                         # Use GPU acceleration
          --threads 32 \                    # Utilize all CPU cores
          --alt-ali 10 \                    # Consider alternative alignments
          --pca 1.0 --pcb 2.0 \            # Conservative pseudocounts for profiles
          --filter-msa 0 \                  # Keep diverse sequences in profile
          --seq-id-mode 1                   # Use shorter seq for %id calculation
        ```
        """
    )
    return


@app.cell
def _(run_command):
    # Execute GPU-accelerated homology search
    run_command(
        [
            "mmseqs", "search",
            "./mmseqs_db/queryDB/queryDB",
            "./mmseqs_db/targetDB/targetDB_gpu",
            "./mmseqs_results/search_results/resultDB",
            "./mmseqs_results/tmp",
            "--comp-bias-corr", "1",
            "-s", "7.5",
            "--max-seqs", "100",
            "--spaced-kmer-mode", "1",
            "--exact-kmer-matching", "0",
            "-a", "1",
            "--alignment-mode", "3",
            "--num-iterations", "3",
            "-e", "1e-20",
            "-c", "0.2",
            "--cov-mode", "5",
            "--realign", "1",
            "--mask", "0",
            "--min-aln-len", "30",
            "--gpu", "1",
            "--threads", "32",
            "--alt-ali", "10",
            "--pca", "1.0",
            "--pcb", "2.0",
            "--e-profile", "1e-20",
            "--filter-msa", "0",
            "--seq-id-mode", "1",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Search on GPU took ~3 minutes""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ##### Results Conversion
        Convert MMseqs2 binary results to tabular format with key metrics:
        """
    )
    return


@app.cell
def _(run_command):
    # Convert binary results to analyzable format
    run_command(
        [
            "mmseqs", "convertalis",
            "./mmseqs_db/queryDB/queryDB",
            "./mmseqs_db/targetDB/targetDB_gpu",
            "./mmseqs_results/search_results/resultDB",
            "./mmseqs_results/search_results/results.m8",
            "--format-output",
            "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov",
        ]
    )
    return


@app.cell
def _(pl):
    column_names = ['query', 'target', 'pident', 'alnlen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits', 'qcov', 'tcov']

    mmseqs_results = pl.read_csv(
        source='./mmseqs_results/search_results/results.m8',
        separator='\t',
        has_header=False)

    mmseqs_results = mmseqs_results.rename({col: name for col, name in zip(mmseqs_results.columns, column_names)})
    mmseqs_results
    return column_names, mmseqs_results


@app.cell
def _(mmseqs_results):
    mmseqs_results.shape
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Sequence Retrieval
        Extract unique protein IDs from search results for subsequent analysis:
        """
    )
    return


@app.cell
def _(mmseqs_results):
    # Efficient ID deduplication with generator expressions
    unique_ids = (
        set(mmseqs_results.get_column('target'))
        .union(set(mmseqs_results.get_column('query')))
    )

    # Write accessions with buffered I/O
    with open("./mmseqs_results/search_results/search_results_accessions.txt", "w") as f:
        for item in unique_ids:
            f.write(f"{item}\n")
    return f, item, unique_ids


@app.cell
def _(run_command):
    # Fast sequence extraction using SeqKit
    run_command(
        [
            "seqkit", "grep",
            "-f", "./mmseqs_results/search_results/search_results_accessions.txt",
            "./mmseqs_db/targetDB/archaea.fa",
            "-o", "./mmseqs_results/search_results/search_results.fasta"
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Protein Embedding Generation
        Leverage ESM-Cambrian-600M model for:

        - Context-aware protein sequence embeddings
        - Sphere normalization for improved clustering
        - Mean pooling over sequence positions
        \"""
        """
    )
    return


@app.cell
def _(bb):
    session = bb.new_session()
    return (session,)


@app.cell
def _(session):
    df = session.read_fasta_file('./mmseqs_results/search_results/search_results.fasta').to_polars()
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### ESM Model Initialization
        Load pre-trained ESM-Cambrian-600M model.
        Model characteristics:

        - Processes sequences up to 1024 residues
        - Generates 1280-dimensional embeddings
        - Captures structural and functional features
        """
    )
    return


@app.cell
def _(ESMC):
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    client.eval()
    return (client,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Embedding Generation Workflow

        1. Generate embeddings via mean-pooled transformer outputs
        2. Apply sphere normalization for directional clustering
        3. Store embeddings as numpy arrays for ML workflows
        """
    )
    return


@app.cell
def _(df, generate_embedding, pl):
    # Batch embedding generation with progress tracking
    df_emb = df.with_columns(
        pl.col("sequence").map_elements(
            generate_embedding,
            return_dtype=pl.Object,
        ).alias("embedding")
    )
    return (df_emb,)


@app.cell
def _(mo):
    mo.md(r"""Running embedding on GPU took 3 minutes""")
    return


@app.cell
def _(df_emb):
    df_emb
    return


@app.cell
def _(cp, df_emb):
    # Convert embeddings to GPU array
    embeddings_cp = cp.asarray(df_emb['embedding'].to_list())
    return (embeddings_cp,)


@app.cell
def _(UMAP, embeddings_cp, perform_dimensionality_reduction):
    # UMAP projection with reproducibility
    reduced = perform_dimensionality_reduction(embeddings_cp,
                                               UMAP,
                                               n_components=3,
                                               metric='euclidean',
                                               n_neighbors=30,
                                              random_state=123)
    return (reduced,)


@app.cell
def _(KMeans, embeddings_cp, perform_clustering):
    # GPU-accelerated K-means clustering
    labels = perform_clustering(embeddings_cp,
                                KMeans,
                                n_clusters=11,
                               random_state=123)
    return (labels,)


@app.cell
def _(create_visualization, labels, reduced):
    # Generate publication-quality visualization
    create_visualization(reduced, labels, "KMeans", "UMAP")
    return


@app.cell
def _(mo):
    mo.md(r"""## AI-Powered Functional Annotation""")
    return


@app.cell
def _(df_emb, labels, pl):
    # Prepare clustered data for AI analysis
    df_emb_clustered = df_emb.with_columns(pl.lit(labels).alias("cluster"))
    df_emb_clustered
    return (df_emb_clustered,)


@app.cell
def _(df_emb_clustered, pl):
    # Cluster size analysis
    df_emb_clustered.group_by("cluster").agg(pl.count()).sort("cluster")
    return


@app.cell
def _(df_emb_clustered, pl):
    # Identify poorly annotated clusters
    pattern = "hypothetical|uncharacterised|duf"

    (df_emb_clustered
        .group_by("cluster")
        .agg([
            pl.col("description").str.to_lowercase()
                .str.contains(pattern).sum().alias("matches"),
            pl.count().alias("total_rows")
        ])
        .with_columns([
            (pl.col("matches") / pl.col("total_rows") * 100).alias("percentage")
        ])
        .sort("percentage", descending=True)
        .head()
    )
    return (pattern,)


@app.cell
def _(mo):
    mo.md(r"""### AI Annotation Pipeline""")
    return


@app.cell
def _(df_emb_clustered, pl):
    df_emb_clustered_annotated = df_emb_clustered. \
        filter(~pl.col('description').str.to_lowercase().str.contains('hypothetical|uncharacterised|duf')). \
        sort('cluster')

    grouped = df_emb_clustered_annotated.group_by("cluster").agg([
        pl.col("description").alias("descriptions")
    ])
    return df_emb_clustered_annotated, grouped


@app.cell
def _(df_emb_clustered_annotated):
    df_emb_clustered_annotated
    return


@app.cell
def _(OpenAI):
    # Initialize Nebius AI Studio client
    client_nebius_ai_studio = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key="<YOUR_API_KEY>"
    )
    return (client_nebius_ai_studio,)


@app.cell
def _(call_ai_api, grouped, tqdm):
    # Process each cluster and collect results
    results = []
    for cluster_id, descriptions in tqdm(grouped.rows(), desc="Analyzing clusters"):
        # Inline f-string prompt construction for cluster analysis
        prompt = (
            f"Below is provided a number of protein descriptions from some cluster of proteins from archaea.\n\n"
            f"Cluster {cluster_id} protein functions analysis:\n"
            f"{chr(10).join(descriptions)}\n\n"
            "Required analysis:\n"
            "1. Primary functional category with confidence score\n"
            "2. Supporting evidence from descriptions\n"
            "3. Immunity system relevance assessment"
        )

        # Using a fixed system role for cluster-level analysis
        system_role = "Senior bioinformatician specializing in archaeal systems"
        functions_summary = call_ai_api(
            model="deepseek-ai/DeepSeek-V3",
            system_role=system_role,
            prompt=prompt,
            max_tokens=8192,
            temperature=0.4,
            top_p=0.95
        )

        results.append({
            "cluster": cluster_id,
            "functions_summary": functions_summary
        })
    return (
        cluster_id,
        descriptions,
        functions_summary,
        prompt,
        results,
        system_role,
    )


@app.cell
def _(pl, results):
    # Convert results to analyzable format
    results_df = pl.DataFrame(results)
    results_df
    return (results_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Integrative Analysis
        Final synthesis steps:

        1. Cross-cluster functional relationships
        2. Immunity system component mapping
        3. Novel defense mechanism identification
        4. Evolutionary conservation patterns
        """
    )
    return


@app.cell
def _(results_df):
    synthesis_prompt = (
        f"Synthesize findings from {len(results_df)} protein clusters:\n"
        f"{results_df.to_dicts()}\n\n"
        "Required analysis:\n"
        "1. Archaeal immune system architecture\n"
        "2. Novel defense mechanism candidates\n"
        "3. Evolutionary conservation patterns\n"
        "4. Proposed experimental validation steps"
    )
    return (synthesis_prompt,)


@app.cell
def _(call_ai_api, synthesis_prompt):
    aggregate_system_role = "Lead researcher in archaeal immunology"
    final_result = call_ai_api(
        model="deepseek-ai/DeepSeek-R1",
        system_role=aggregate_system_role,
        prompt=synthesis_prompt,
        max_tokens=8192,
        temperature=0.3
    )
    return aggregate_system_role, final_result


@app.cell
def _(final_result):
    final_result
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(Path):
    def save_to_fasta(data, output_file):
        if not data or not isinstance(data, list):
            raise ValueError("Input data must be a non-empty list of dictionaries.")

        # Create a Path object from the output_file
        output_path = Path(output_file)

        # Create the parent directory if it doesn't exist
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_path.parent}")

        with output_path.open('w') as fasta_file:
            for entry in data:
                entry_id = entry.get('Entry')
                sequence = entry.get('Sequence')
                if not entry_id or not sequence:
                    raise ValueError("Each dictionary must contain 'Entry' and 'Sequence' keys with non-empty values.")
                fasta_file.write(f">{entry_id}\n{sequence}\n")
    return (save_to_fasta,)


@app.cell
def _(Optional, Path, subprocess):
    def run_command(
        command: list[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None
    ) -> None:
        try:
            # Execute the command
            result = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                timeout=timeout,
                check=True,          # Raise an exception for non-zero exit codes
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                text=True            # Decode output as text
            )

            # Print the standard output
            if result.stdout:
                print(result.stdout)

            # Optionally, print the standard error
            if result.stderr:
                print(result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"Error: Command '{' '.join(command)}' failed with exit code {e.returncode}")
            print(f"stderr: {e.stderr}")
            raise e  # Re-raise the exception if you want to handle it further up
        except FileNotFoundError:
            print(f"Error: Command not found: {command[0]}")
            raise
        except subprocess.TimeoutExpired:
            print(f"Error: Command '{' '.join(command)}' timed out after {timeout} seconds")
            raise
    return (run_command,)


@app.cell
def _(ESMProtein, LogitsConfig, client, np, torch):
    # Optimized embedding generator with GPU memory management
    def generate_embedding(sequence: str):
        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                encoded = client.encode(ESMProtein(sequence=sequence))
                logits = client.logits(encoded, LogitsConfig(return_embeddings=True))
                seq_embeddings = logits.embeddings[0, 1:-1, :]

                # Memory-efficient operations
                embedding = torch.mean(seq_embeddings, dim=0)
                embedding = sphere_normalize(embedding)

                return embedding.cpu().numpy().squeeze()
            except RuntimeError as e:
                print(f"Error processing sequence: {str(e)}")
                return np.nan

    # Numerical stable normalization
    def sphere_normalize(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + epsilon)
    return generate_embedding, sphere_normalize


@app.cell
def _(cp, np, pl, plt):
    # Comprehensive clustering and visualization suite
    def perform_clustering(embeddings, algorithm, **cluster_kwargs):
        """GPU-accelerated clustering with automatic type handling"""
        if isinstance(embeddings, pl.DataFrame):
            embeddings = embeddings.to_numpy()

        cluster_model = algorithm(**cluster_kwargs)

        if hasattr(cluster_model, 'fit_predict'):
            labels = cluster_model.fit_predict(embeddings)
        else:
            cluster_model.fit(embeddings)
            labels = cluster_model.labels_

        return cp.asnumpy(labels) if isinstance(labels, cp.ndarray) else labels

    def perform_dimensionality_reduction(embeddings, algorithm, **dr_kwargs):
        """Dimensionality reduction with GPU/CPU dispatch"""
        if isinstance(embeddings, pl.DataFrame):
            embeddings = embeddings.to_numpy()

        dr_model = algorithm(**dr_kwargs)
        reduced = dr_model.fit_transform(embeddings)
        return cp.asnumpy(reduced) if isinstance(reduced, cp.ndarray) else reduced

    def create_visualization(reduced_embeddings, labels,
                            clustering_algo_name="Clustering",
                            dr_algo_name="DR",
                            figsize=(16, 16)):
        """Publication-quality 3D/2D visualization"""
        plt.figure(figsize=figsize, facecolor='white')
        fig = plt.gcf()
        fig.suptitle(f"{dr_algo_name} Projection Colored by {clustering_algo_name}", y=1.02)

        # Plot configuration
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224, projection='3d')

        # Color mapping
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab20')
        color_map = {
            label: (0.5, 0.5, 0.5, 1) if label == -1 else cmap(i % 20)
            for i, label in enumerate(unique_labels)
        }
        point_colors = [color_map[label] for label in labels]

        # 2D projections
        projection_pairs = [(0, 1), (0, 2), (1, 2)]
        for ax, (x, y) in zip([ax1, ax2, ax3], projection_pairs):
            ax.scatter(
                reduced_embeddings[:, x],
                reduced_embeddings[:, y],
                c=point_colors,
                s=12,
                alpha=0.7,
                edgecolors='w',
                linewidths=0.2
            )
            ax.set_xlabel(f"{dr_algo_name} {x+1}", fontsize=9)
            ax.set_ylabel(f"{dr_algo_name} {y+1}", fontsize=9)
            ax.grid(True, alpha=0.2)

        # 3D visualization
        ax4.scatter3D(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=point_colors,
            s=12,
            alpha=0.7,
            depthshade=True
        )
        ax4.set_xlabel(f"{dr_algo_name} 1", fontsize=9)
        ax4.set_ylabel(f"{dr_algo_name} 2", fontsize=9)
        ax4.set_zlabel(f"{dr_algo_name} 3", fontsize=9)

        # Legend
        legend_elements = [
            plt.Line2D([], [], marker='o', linestyle='',
                      color=color_map[label],
                      label=f"Cluster {label}" if label != -1 else "Noise")
            for label in unique_labels
        ]

        ax4.legend(
            handles=legend_elements,
            title="Clusters",
            bbox_to_anchor=(1.15, 1),
            loc='upper right',
            fontsize=8
        )

        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    return (
        create_visualization,
        perform_clustering,
        perform_dimensionality_reduction,
    )


@app.cell
def _(client_nebius_ai_studio):
    def call_ai_api(model, system_role, prompt, max_tokens=8192, temperature=0.4, top_p=0.95):
        """
        Call the AI API with specified parameters.

        :param model: Model identifier.
        :param system_role: Content for the system message.
        :param prompt: User prompt.
        :param max_tokens: Maximum token limit.
        :param temperature: Sampling temperature.
        :param top_p: Top-p nucleus sampling probability.
        :return: API response string or error message.
        """
        try:
            response = client_nebius_ai_studio.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"
    return (call_ai_api,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()