"""Generate a large dataset of 5000 ER graphs."""
import networkx as nx
import random
from pathlib import Path

def generate_er_graphs(num_graphs=5000, output_dir="data/large_5000"):
    """Generate random ER graphs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_graphs} ER graphs...")

    for i in range(num_graphs):
        # Random number of nodes between 10-20
        n = random.randint(10, 20)
        # Edge probability
        p = 0.3

        # Generate ER graph
        G = nx.erdos_renyi_graph(n, p)

        # Save as GraphML
        output_file = output_path / f"graph_{i:05d}.graphml"
        nx.write_graphml(G, output_file)

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{num_graphs} graphs...")

    print(f"✓ Generated {num_graphs} graphs in {output_dir}")

if __name__ == "__main__":
    generate_er_graphs()
