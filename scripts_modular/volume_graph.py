#!/usr/bin/env python3
"""
Volume Dependency Graph Manager

Manages the dependency graph for multi-volume panoramic stitching.
Determines execution order and identifies parallel processing opportunities.

Features:
- Load configuration from volume_layout.json
- Build directed acyclic graph (DAG) of volume dependencies
- Topological sort for execution order
- Level-based grouping for parallel processing
- Validation of graph structure

Author: OCT Panoramic Stitching System
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class VolumeNode:
    """Represents a single volume in the dependency graph."""

    def __init__(self, volume_id: int, config: dict):
        """
        Initialize a volume node.

        Args:
            volume_id: Numeric ID of the volume
            config: Configuration dictionary from volume_layout.json
        """
        self.id = volume_id
        self.position = config.get('position', 'unknown')
        self.folder = config.get('folder', f'folder_{volume_id}')
        self.neighbor = config.get('neighbor', None)  # Reference volume for stitching
        self.level = config.get('level', 0 if volume_id == 1 else None)
        self.role = config.get('role', 'moving' if volume_id != 1 else 'reference')
        self.description = config.get('description', '')

        # Graph relationships
        self.dependencies: Set[int] = set()  # Volumes this one depends on
        self.dependents: Set[int] = set()     # Volumes that depend on this one

    def add_dependency(self, volume_id: int):
        """Add a dependency (this volume needs volume_id to be processed first)."""
        self.dependencies.add(volume_id)

    def add_dependent(self, volume_id: int):
        """Add a dependent (volume_id needs this volume to be processed first)."""
        self.dependents.add(volume_id)

    def __repr__(self):
        return f"VolumeNode(id={self.id}, position={self.position}, level={self.level})"


class VolumeGraph:
    """
    Directed acyclic graph (DAG) for managing volume stitching dependencies.

    Provides:
    - Dependency resolution
    - Parallel execution groups
    - Topological ordering
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the volume graph.

        Args:
            config_path: Path to volume_layout.json (default: auto-detect)
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'volume_layout.json'

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.nodes: Dict[int, VolumeNode] = {}
        self.levels: Dict[int, List[int]] = defaultdict(list)  # level -> [volume_ids]

        # Build graph
        self._build_graph()
        self._validate_graph()

        logger.info(f"Volume graph initialized with {len(self.nodes)} volumes")

    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _build_graph(self):
        """Build the dependency graph from configuration."""
        volumes_config = self.config.get('volumes', {})

        # Create nodes
        for vol_id_str, vol_config in volumes_config.items():
            vol_id = int(vol_id_str)
            node = VolumeNode(vol_id, vol_config)
            self.nodes[vol_id] = node

            # Group by level
            if node.level is not None:
                self.levels[node.level].append(vol_id)

        # Build edges (dependencies)
        for vol_id, node in self.nodes.items():
            if node.neighbor is not None:
                # This volume depends on its neighbor
                node.add_dependency(node.neighbor)
                self.nodes[node.neighbor].add_dependent(vol_id)

                logger.debug(f"Volume {vol_id} depends on volume {node.neighbor}")

        # Sort levels by key
        self.levels = dict(sorted(self.levels.items()))

        logger.info(f"Graph structure: {len(self.levels)} levels")
        for level, vol_ids in self.levels.items():
            logger.info(f"  Level {level}: volumes {vol_ids}")

    def _validate_graph(self):
        """
        Validate the graph structure.

        Checks:
        - No cycles (DAG property)
        - All dependencies exist
        - Reference volume (id=1) has no dependencies
        """
        # Check reference volume
        if 1 not in self.nodes:
            raise ValueError("Reference volume (id=1) not found in configuration")

        if len(self.nodes[1].dependencies) > 0:
            raise ValueError("Reference volume (id=1) should have no dependencies")

        # Check all dependencies exist
        for vol_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(f"Volume {vol_id} depends on non-existent volume {dep_id}")

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: int) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dependent_id in self.nodes[node_id].dependents:
                if dependent_id not in visited:
                    if has_cycle(dependent_id):
                        return True
                elif dependent_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for vol_id in self.nodes:
            if vol_id not in visited:
                if has_cycle(vol_id):
                    raise ValueError("Cycle detected in dependency graph")

        logger.info("Graph validation passed ✓")

    def get_execution_levels(self) -> List[List[int]]:
        """
        Get volumes grouped by execution level for parallel processing.

        Returns:
            List of levels, where each level is a list of volume IDs that can be
            processed in parallel.

        Example:
            [[1], [2, 4, 6, 8], [3, 5, 7, 9]]
        """
        return [self.levels[level] for level in sorted(self.levels.keys())]

    def get_processing_pairs(self, level: int) -> List[Tuple[int, int]]:
        """
        Get (reference, moving) pairs for a specific level.

        Args:
            level: Processing level (1, 2, ...)

        Returns:
            List of (reference_id, moving_id) tuples
        """
        pairs = []

        if level not in self.levels:
            return pairs

        for moving_id in self.levels[level]:
            node = self.nodes[moving_id]
            if node.neighbor is not None:
                pairs.append((node.neighbor, moving_id))

        return pairs

    def get_all_processing_pairs(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get all processing pairs grouped by level.

        Returns:
            Dictionary mapping level -> [(reference_id, moving_id), ...]
        """
        all_pairs = {}
        for level in sorted(self.levels.keys()):
            if level > 0:  # Skip level 0 (reference volume)
                all_pairs[level] = self.get_processing_pairs(level)
        return all_pairs

    def get_topological_order(self) -> List[int]:
        """
        Get volumes in topological order using Kahn's algorithm.

        Returns:
            List of volume IDs in valid execution order
        """
        # Calculate in-degrees
        in_degree = {vol_id: len(node.dependencies) for vol_id, node in self.nodes.items()}

        # Queue of nodes with no dependencies
        queue = deque([vol_id for vol_id, degree in in_degree.items() if degree == 0])

        topo_order = []

        while queue:
            vol_id = queue.popleft()
            topo_order.append(vol_id)

            # Reduce in-degree for dependents
            for dependent_id in self.nodes[vol_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(topo_order) != len(self.nodes):
            raise ValueError("Graph contains cycle - topological sort failed")

        return topo_order

    def get_subsampling_config(self) -> dict:
        """Get subsampling configuration from config."""
        return self.config.get('subsampling', {})

    def get_output_config(self) -> dict:
        """Get output configuration from config."""
        return self.config.get('output', {})

    def get_volume_folder(self, volume_id: int) -> str:
        """Get the folder name for a specific volume."""
        if volume_id not in self.nodes:
            raise ValueError(f"Volume {volume_id} not found in graph")
        return self.nodes[volume_id].folder

    def print_graph_summary(self):
        """Print a summary of the graph structure."""
        print("\n" + "="*70)
        print("VOLUME DEPENDENCY GRAPH SUMMARY")
        print("="*70)

        print(f"\nTotal volumes: {len(self.nodes)}")
        print(f"Processing levels: {len([l for l in self.levels if l > 0])}")

        # Subsampling info
        subsample_config = self.get_subsampling_config()
        if subsample_config.get('enabled', False):
            print(f"Subsampling: Every {subsample_config.get('z_stride', 1)}th B-scan")

        print("\nExecution Levels:")
        for level, vol_ids in sorted(self.levels.items()):
            print(f"  Level {level}: {len(vol_ids)} volumes - {vol_ids}")

        print("\nProcessing Pairs:")
        for level, pairs in self.get_all_processing_pairs().items():
            print(f"  Level {level} (parallel):")
            for ref_id, mov_id in pairs:
                ref_node = self.nodes[ref_id]
                mov_node = self.nodes[mov_id]
                print(f"    Volume {mov_id} ({mov_node.position}) → Volume {ref_id} ({ref_node.position})")

        print("\nTopological Order:")
        topo_order = self.get_topological_order()
        print(f"  {' → '.join(map(str, topo_order))}")

        print("\n" + "="*70)


def main():
    """Test the volume graph functionality."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create graph
    graph = VolumeGraph()

    # Print summary
    graph.print_graph_summary()

    # Test specific queries
    print("\n" + "="*70)
    print("TESTING GRAPH QUERIES")
    print("="*70)

    print("\nLevel 1 processing pairs:")
    for ref_id, mov_id in graph.get_processing_pairs(1):
        print(f"  {ref_id} ← {mov_id}")

    print("\nLevel 2 processing pairs:")
    for ref_id, mov_id in graph.get_processing_pairs(2):
        print(f"  {ref_id} ← {mov_id}")

    print("\nVolume 5 dependencies:")
    print(f"  Depends on: {graph.nodes[5].dependencies}")
    print(f"  Needed by: {graph.nodes[5].dependents}")


if __name__ == "__main__":
    main()
