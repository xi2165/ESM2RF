import os
import json
import subprocess
import tempfile
import shutil
import numpy as np
import networkx as nx
import pandas as pd
from config import NumpyEncoder

def create_protein_similarity_graph(protein_pairs, output_dir="./output", min_bitscore=50):
    """
    Create protein similarity graph using MMseqs2 for sequence similarity calculation
    
    Return:
        nx.Graph: Protein similarity graph
        protein_index : Dict[str, int]
        all_proteins : List[str] <-unique protein
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract all unique protein sequences
    all_proteins = []
    protein_index = {}  # protein seq -> protein index
    
    for seq in set([p['seq1'] for p in protein_pairs] + [p['seq2'] for p in protein_pairs]):
        protein_index[seq] = len(all_proteins)
        all_proteins.append(seq)
    
    print(f"Extracted {len(all_proteins)} unique protein sequences from {len(protein_pairs)} protein pairs")
    
    # Create fasta file
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "proteins.fasta")
    
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(all_proteins):
            f.write(f">protein_{i}\n{seq}\n")
    
    try:
        # Run mmseqs2 to calculate sequence similarity
        db_path = os.path.join(temp_dir, "db")
        result_db = os.path.join(temp_dir, "result_db")
        result_path = os.path.join(output_dir, "similarity.tsv")
        tmp_path = os.path.join(temp_dir, "tmp")
        
        # Create temporary working directory
        os.makedirs(tmp_path, exist_ok=True)
        
        # Create database
        subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True)
        
        # Execute sequence search to calculate similarity
        subprocess.run([
            "mmseqs", "search", db_path, db_path, result_db, tmp_path,
            "-s", "7.5", "--max-seqs", "1000"
            ], check=True)
        
        # Output results including bitscore
        subprocess.run([
            "mmseqs", "convertalis", db_path, db_path, result_db, result_path,
            "--format-output", "query,target,bits"
            ], check=True)
        
        # Build similarity graph
        G = nx.Graph()
        
        # Add all protein nodes
        for i, seq in enumerate(all_proteins):
            G.add_node(i, sequence=seq)
        
        # Add similarity edges
        with open(result_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = int(parts[0].split('_')[1])
                    target_id = int(parts[1].split('_')[1])
                    bitscore = float(parts[2])
                    
                    # Only add edges above threshold
                    if bitscore >= min_bitscore and query_id != target_id:
                        G.add_edge(query_id, target_id, weight=bitscore)
        
        print(f"Similarity graph construction completed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G, protein_index, all_proteins
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)

def save_partition_files(partition, categories, protein_index, all_proteins, output_dir):
    """Save partition results to files"""
    # Save partition results as JSON
    partition_file = os.path.join(output_dir, "partition_results.json")
    
    data_to_save = {
        'partition': partition,
        'protein_index': protein_index,
        'all_proteins': all_proteins
    }
    
    with open(partition_file, 'w') as f:
        json.dump(data_to_save, f, cls=NumpyEncoder)
    
    # Save protein partition CSV
    partition_df = pd.DataFrame({
        'protein_idx': list(partition.keys()),
        'partition': list(partition.values()),
        'sequence': [all_proteins[idx] for idx in partition.keys()]
    })
    partition_df.to_csv(os.path.join(output_dir, "protein_partition.csv"), index=False)
    
    # Save pair categories CSV
    categories_df = pd.DataFrame({
        'pair_idx': list(categories.keys()),
        'category': list(categories.values()),
        'category_name': [['INTRA₀', 'INTRA₁', 'INTER'][cat] for cat in categories.values()]
    })
    categories_df.to_csv(os.path.join(output_dir, "pair_categories.csv"), index=False)
    
    print(f"Partition results saved to: {partition_file}")

def load_partition_results(output_dir="./output"):
    """
    Load partition results from file
    """
    partition_file = os.path.join(output_dir, "partition_results.json")
    
    if not os.path.exists(partition_file):
        print(f"Saved partition results not found: {partition_file}")
        return None, None, None, False
    
    try:
        with open(partition_file, 'r') as f:
            data = json.load(f)
        
        # string keys -> integer keys
        partition = {int(k): v for k, v in data['partition'].items()}
        protein_index = data['protein_index']
        all_proteins = data['all_proteins']
        
        print(f"Successfully loaded partition results: {partition_file}")
        
        # Count nodes in each partition
        partition_sizes = {}
        for node, part in partition.items():
            if part not in partition_sizes:
                partition_sizes[part] = 0
            partition_sizes[part] += 1
        
        print("Partition results:")
        for part, size in partition_sizes.items():
            print(f"  Partition {part}: {size} proteins")
        
        return partition, protein_index, all_proteins, True
    
    except Exception as e:
        print(f"Error loading partition results: {e}")
        return None, None, None, False

def partition_proteins_kahip(G, output_dir="./output", k=2, force_recompute=False):
    """
    Perform graph partitioning using KaHIP KaFFPa, or load from saved results
    """
    # Check if saved partition results exist
    if not force_recompute:
        partition, _, _, success = load_partition_results(output_dir)
        if success:
            return partition
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Graph -> KaHIP format
    graph_file = os.path.join(output_dir, "graph.kahip")
    
    # KaHIP format: first line "num_nodes num_edges [weight_type]"
    # weight_type 1 means edges have weights
    with open(graph_file, 'w') as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()} 1\n")
        
        # For each node, write its edges
        for node in range(G.number_of_nodes()):
            edges = []
            for neighbor, data in G[node].items():
                weight = int(data['weight'])  # KaHIP requires integer weights
                edges.append(f"{neighbor+1} {weight}")  # KaHIP node numbering starts from 1
            
            f.write(" ".join(edges) + "\n")
    
    # Run KaHIP KaFFPa
    partition_file = os.path.join(output_dir, "partition.kahip")
    
    # KaFFPa: minimize cut edge weight sum
    subprocess.run([
        "kaffpa", graph_file, 
        "--k", str(k),
        "--preconfiguration=strong",
        "--output_filename", partition_file
    ], check=True)
    
    # Read partition results
    partition = {}
    
    with open(partition_file, 'r') as f:
        for i, line in enumerate(f):
            partition[i] = int(line.strip())
    
    # Count nodes in each partition
    partition_sizes = {}
    for node, part in partition.items():
        if part not in partition_sizes:
            partition_sizes[part] = 0
        partition_sizes[part] += 1
    
    print("Partition results:")
    for part, size in partition_sizes.items():
        print(f"  Partition {part}: {size} proteins")
    
    return partition

def categorize_protein_pairs(protein_pairs, partition, protein_index):
    """
    Categorize protein pairs based on partition results
    """
    categories = {}
    
    for i, pair in enumerate(protein_pairs):
        seq1, seq2 = pair['seq1'], pair['seq2']
        
        # Get protein indices in graph
        idx1 = protein_index[seq1]
        idx2 = protein_index[seq2]
        
        # Get partition information
        part1 = partition[idx1]
        part2 = partition[idx2]
        
        # Classification rules
        if part1 == part2:
            if part1 == 0:
                categories[i] = 0  # INTRA₀
            else:
                categories[i] = 1  # INTRA₁
        else:
            categories[i] = 2  # INTER
    
    return categories

def print_experiment_statistics(splits, y):
    """
    Print statistics for all experiments
    """
    print("\n===== Experiment Design Statistics =====")
    
    experiment_names = {
        'exp1': 'Experiment 1 (Train: INTRA₀, Test: INTRA₁)',
        'exp2': 'Experiment 2 (Train: INTER, Test: INTRA₀)', 
        'exp3': 'Experiment 3 (Train: INTER, Test: INTRA₁)',
        'exp4': 'Experiment 4 (Random 80/20 split)'
    }
    
    for exp_name, description in experiment_names.items():
        if exp_name in splits:
            train_idx, test_idx = splits[exp_name]
            train_pos = sum(y[train_idx] == 1)
            test_pos = sum(y[test_idx] == 1)
            
            print(f"\n{description}:")
            print(f"  Training set: {len(train_idx)} samples (positive: {train_pos})")
            print(f"  Test set: {len(test_idx)} samples (positive: {test_pos})")

def kahip_split(X, y, protein_pairs, output_dir="./output", min_bitscore=50, force_recompute=False, use_global_partition=True):
    """
    Create training/test set splits based on graph partitioning
    
    Args:
        use_global_partition: Whether to prioritize using global fixed partitions (default True)
    """
    # Initialize success variable
    success = False
    
    # If using global partitions, prioritize loading from project root output/
    if use_global_partition and not force_recompute:
        global_output_dir = "./output"  # Project root output directory
        partition, protein_index, all_proteins, success = load_partition_results(global_output_dir)
        
        if success:
            print(f"Using saved partition results: {global_output_dir}/partition_results.json")
            # Continue using loaded partitions for subsequent processing
        else:
            print(f"Saved partition results not found: {global_output_dir}/partition_results.json")
            print("Hint: Run 'python split.py' to generate fixed partitions")
            # If global partitions not found, continue with original logic
    
    if not success or force_recompute:
        # If loading failed or forced recomputation, build protein similarity graph
        print("Step 1: Building protein similarity graph...")
        G, protein_index, all_proteins = create_protein_similarity_graph(
            protein_pairs, output_dir=output_dir, min_bitscore=min_bitscore
        )
        
        # Use KaHIP for graph partitioning
        print("Step 2: Using KaHIP for graph partitioning...")
        partition = partition_proteins_kahip(G, output_dir=output_dir, k=2, force_recompute=force_recompute)
    else:
        print("Using saved partition results...")
        # If successfully loaded partition results, no need to rebuild graph and compute partitions
    
    # Categorize protein pairs
    print("Step 3: Categorizing protein pairs...")
    categories = categorize_protein_pairs(protein_pairs, partition, protein_index)
    
    # Create training/test sets for three experimental designs
    print("Step 4: Creating experimental designs...")
    
    # Get sample indices for each category
    intra0_indices = np.array([i for i, cat in categories.items() if cat == 0])
    intra1_indices = np.array([i for i, cat in categories.items() if cat == 1])
    inter_indices = np.array([i for i, cat in categories.items() if cat == 2])
    
    # Experimental design 1: Train on INTRA₀, test on INTRA₁
    exp1_train_idx = intra0_indices
    exp1_test_idx = intra1_indices
    
    # Experimental design 2: Train on INTER, test on INTRA₀
    exp2_train_idx = inter_indices
    exp2_test_idx = intra0_indices
    
    # Experimental design 3: Train on INTER, test on INTRA₁
    exp3_train_idx = inter_indices
    exp3_test_idx = intra1_indices

    # Experimental design 4: 8:2 random split (baseline comparison)
    np.random.seed(42)  # Set random seed for reproducibility
    all_indices = np.arange(len(y))
    np.random.shuffle(all_indices)
    split_point = int(0.8 * len(all_indices))
    exp4_train_idx = all_indices[:split_point]
    exp4_test_idx = all_indices[split_point:]

    # Create splits dictionary
    splits = {
        'exp1': (exp1_train_idx, exp1_test_idx),
        'exp2': (exp2_train_idx, exp2_test_idx),
        'exp3': (exp3_train_idx, exp3_test_idx),
        'exp4': (exp4_train_idx, exp4_test_idx),
        'partition': partition,
        'categories': categories,
        'protein_index': protein_index,
        'all_proteins': all_proteins
    }
    
    # Print experiment statistics
    print_experiment_statistics(splits, y)
    
    if not success or force_recompute:
        save_partition_files(partition, categories, protein_index, all_proteins, output_dir)
    else:
        print(f"Skipping partition file save - using global partition from ./output/")
    
    return splits

def generate_fixed_partitions(data_file="database.txt", output_dir="./output", min_bitscore=50):
    """
    Generate fixed partition results
    """
    print("=== Generating Fixed Partition Results ===")
    
    from data import load_data
    
    # Load data
    print("Loading data...")
    all_data = load_data(data_file)
    print(f"Data loading completed: {len(all_data)} samples")
    
    # Prepare data
    protein_pairs = all_data
    y = np.array([item['interaction'] for item in all_data])
    X = None  # kahip_split doesn't need X
    
    # Generate partitions (force recomputation)
    print("Generating protein partitions...")
    splits = kahip_split(X, y, protein_pairs, 
                        output_dir=output_dir, 
                        min_bitscore=min_bitscore, 
                        force_recompute=True)
    
    print("\n=== Partition Generation Completed ===")
    print(f"Partition results saved to: {output_dir}/partition_results.json")
    print(f"Protein partition info: {output_dir}/protein_partition.csv")
    print(f"Protein pair classification: {output_dir}/pair_categories.csv")
    print(f"Similarity data: {output_dir}/similarity.tsv")
    
    # Print partition statistics
    print("\n=== Partition Statistics ===")
    for exp_name, split_data in splits.items():
        if exp_name.startswith('exp'):  # Only process experiment partitions
            train_idx, test_idx = split_data
            train_pos = sum(y[train_idx] == 1)
            test_pos = sum(y[test_idx] == 1)
            print(f"{exp_name}: Training set {len(train_idx)} samples (positive: {train_pos}), "
                  f"Test set {len(test_idx)} samples (positive: {test_pos})")
    
    return splits

def main():
    """
    Command line entry function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fixed protein partition results')
    parser.add_argument('--data_file', '-d', default='database.txt', 
                       help='Data file path (default: database.txt)')
    parser.add_argument('--output_dir', '-o', default='./output', 
                       help='Output directory (default: ./output)')
    parser.add_argument('--min_bitscore', '-b', type=int, default=50, 
                       help='Minimum bitscore threshold (default: 50)')
    
    args = parser.parse_args()
    
    # Generate fixed partitions
    generate_fixed_partitions(
        data_file=args.data_file,
        output_dir=args.output_dir,
        min_bitscore=args.min_bitscore
    )

if __name__ == "__main__":
    main()