import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# CPU-Based Bioinfromatics Data Processing""")
    return


@app.cell
def _():
    from pprint import pprint
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Optional
    import subprocess

    import marimo as mo
    import polars as pl
    import biobear as bb
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.cluster import KMeans
    from umap import UMAP
    return (
        Axes3D,
        ESMC,
        ESMProtein,
        F,
        KMeans,
        LogitsConfig,
        Optional,
        Path,
        UMAP,
        bb,
        dataclass,
        mcolors,
        mo,
        nn,
        np,
        pl,
        plt,
        pprint,
        subprocess,
        torch,
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
    uniprot_arch_defense_proteins_df = pl.read_csv("https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Cxref_string%2Cxref_interpro%2Corganism_id%2Csequence%2Cxref_pdb%2Ccc_function&format=tsv&query=%28%28taxonomy_id%3A2157%29+AND+%28%28cc_function%3Aimmun*%29+OR+%28cc_function%3Adefens*%29+OR+%28cc_function%3Aantimicrob*%29+OR+%28cc_function%3Aantibacter*%29%29%29", separator='\t')
    return (uniprot_arch_defense_proteins_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Initial Dataset Inspection
        First look at the retrieved dataset structure and contents.
        """
    )
    return


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
        """
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
          --gpu 0 \                         # Disable GPU
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
    run_command(
        [
            "mmseqs", "search",
            "./mmseqs_db/queryDB/queryDB",
            "./mmseqs_db/targetDB/targetDB",
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
            "--gpu", "0",
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
    mo.md(r"""Search on CPU took 13 minutes""")
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
    run_command(
        [
            "mmseqs", "convertalis",
            "./mmseqs_db/queryDB/queryDB",
            "./mmseqs_db/targetDB/targetDB",
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
    mmseqs_results.head()
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
    unqiue_ids = set(mmseqs_results.get_column('target').to_list()) | set(mmseqs_results.get_column('query').to_list())
    with open("./mmseqs_results/search_results/search_results_accessions.txt", "w") as f:
        for item in unqiue_ids:
            f.write(item + "\n")
    return f, item, unqiue_ids


@app.cell
def _(mo):
    mo.md(r"""### Fast Sequence Extraction with SeqKit""")
    return


@app.cell
def _(run_command):
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
    client = ESMC.from_pretrained("esmc_600m").to("cpu")
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
    df_emb = df.with_columns(
        pl.col("sequence").map_elements(
            generate_embedding,
            return_dtype=pl.Object,
        ).alias("embedding")
    )
    return (df_emb,)


@app.cell
def _(df_emb):
    df_emb.head()
    return


@app.cell
def _(mo):
    mo.md(r"""Running embedding on CPU took 54 minutes""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Dimensionality Reduction & Clustering
        Analytical pipeline:

        1. **UMAP**: Non-linear dimensionality reduction to 3D
        2. **K-means**: Partition embeddings into functional clusters
        3. **Visualization**: 2D/3D plots colored by cluster membership
        """
    )
    return


@app.cell
def _(df_emb, np):
    embeddings_np = np.asarray(df_emb['embedding'].to_list())
    return (embeddings_np,)


@app.cell
def _(KMeans, embeddings_np, perform_clustering):
    # Clustering
    labels = perform_clustering(embeddings_np,
                                KMeans,
                                n_clusters=11,
                               random_state=123)
    return (labels,)


@app.cell
def _(UMAP, embeddings_np, perform_dimensionality_reduction):
    # Dimensionality reduction
    reduced = perform_dimensionality_reduction(embeddings_np,
                                               UMAP,
                                               n_components=3,
                                               metric='euclidean',
                                               n_neighbors=30,
                                              random_state=123)
    return (reduced,)


@app.cell
def _(mo):
    mo.md(r"""Running UMAP on CPU took 11 seconds""")
    return


@app.cell
def _(create_visualization, labels, reduced):
    # Visualization
    create_visualization(reduced, labels, "KMeans", "UMAP")
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
def _(ESMProtein, LogitsConfig, client, torch):
    def generate_embedding(sequence: str):
        with torch.no_grad():
            # Encode sequence
            encoded = client.encode(ESMProtein(sequence=sequence))
            logits = client.logits(
                encoded,
                LogitsConfig(return_embeddings=True)
            )
            # Get embeddings
            seq_embeddings = logits.embeddings[0, 1:-1, :]  # [seq_len, emb_size]
            embedding = torch.mean(seq_embeddings, dim=0)


            # Apply sphere normalization
            emb_normalized = sphere_normalize(embedding)

            # Convert to numpy
            return emb_normalized.float().numpy().squeeze()

    def sphere_normalize(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Sphere normalization with numerical stability"""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + epsilon)
    return generate_embedding, sphere_normalize


@app.cell
def _(np, pl, plt):
    def perform_clustering(embeddings, algorithm, **cluster_kwargs):
        """
        Perform clustering on embeddings using specified algorithm

        Args:
            embeddings: Input embeddings (cuPy array or Polars DataFrame)
            algorithm: Clustering algorithm class (e.g., cuml.KMeans)
            **cluster_kwargs: Keyword arguments for clustering algorithm

        Returns:
            Cluster labels as numpy array
        """
        if isinstance(embeddings, pl.DataFrame):
            embeddings = embeddings.to_numpy()

        cluster_model = algorithm(**cluster_kwargs)

        if hasattr(cluster_model, 'fit_predict'):
            labels = cluster_model.fit_predict(embeddings)
        else:
            cluster_model.fit(embeddings)
            labels = cluster_model.labels_

        return labels

    def perform_dimensionality_reduction(embeddings, algorithm, **dr_kwargs):
        """
        Perform dimensionality reduction on embeddings

        Args:
            embeddings: Input embeddings (cuPy array or Polars DataFrame)
            algorithm: DR algorithm class (e.g., cuml.UMAP)
            **dr_kwargs: Keyword arguments for DR algorithm

        Returns:
            Reduced embeddings as numpy array
        """
        if isinstance(embeddings, pl.DataFrame):
            embeddings = embeddings.to_numpy()

        dr_model = algorithm(**dr_kwargs)
        reduced = dr_model.fit_transform(embeddings)
        return reduced

    def create_visualization(reduced_embeddings, labels,
                            clustering_algo_name="Clustering",
                            dr_algo_name="DR",
                            figsize=(12, 12)):
        """
        Create 2x2 visualization grid with 3D and 2D projections

        Args:
            reduced_embeddings: Reduced embeddings with >=3 dimensions (numpy array)
            labels: Cluster labels (numpy array)
            clustering_algo_name: Name of clustering algorithm for titles
            dr_algo_name: Name of DR algorithm for titles
            figsize: Figure size
        """
        if reduced_embeddings.shape[1] < 3:
            raise ValueError("Reduced embeddings must have at least 3 dimensions")

        plt.figure(figsize=figsize, facecolor='white')
        fig = plt.gcf()
        fig.suptitle(f"{dr_algo_name} Projection Colored by {clustering_algo_name}", y=1.02)

        # Create plot grid
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224, projection='3d')

        # Color setup
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab10')
        color_map = {
            label: (0, 0, 0, 1) if label == -1 else cmap(i % 10)
            for i, label in enumerate(unique_labels)
        }
        point_colors = [color_map[label] for label in labels]

        # 2D Projections
        for ax, (x, y) in zip([ax1, ax2, ax3], [(0, 1), (0, 2), (1, 2)]):
            ax.scatter(
                reduced_embeddings[:, x],
                reduced_embeddings[:, y],
                c=point_colors,
                s=8,
                alpha=0.6
            )
            ax.set_xlabel(f"{dr_algo_name} Dim {x+1}")
            ax.set_ylabel(f"{dr_algo_name} Dim {y+1}")
            ax.set_facecolor('white')
            ax.grid(True, alpha=0.3)

        # 3D Projection
        ax4.scatter3D(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=point_colors,
            s=8,
            alpha=0.6,
            depthshade=False
        )
        ax4.set_xlabel(f"{dr_algo_name} Dim 1")
        ax4.set_ylabel(f"{dr_algo_name} Dim 2")
        ax4.set_zlabel(f"{dr_algo_name} Dim 3")
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False

        # Legend
        legend_handles = [
            plt.Line2D([], [], marker='o', linestyle='',
                       color=color_map[label],
                       label="Noise" if label == -1 else f"Cluster {label}")
            for label in unique_labels
        ]

        ax4.legend(
            handles=legend_handles,
            title=f"{clustering_algo_name} Clusters",
            bbox_to_anchor=(1.1, 1),
            loc='upper left',
            fontsize='small'
        )

        plt.tight_layout()
        plt.savefig('cluster_DR.png')
        plt.show()
    return (
        create_visualization,
        perform_clustering,
        perform_dimensionality_reduction,
    )


if __name__ == "__main__":
    app.run()