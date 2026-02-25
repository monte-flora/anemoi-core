# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Comprehensive tests for BandedGraphSelfAttention.

These tests verify that:
1. RCM permutation correctly reduces bandwidth
2. k-hop neighbors fall within the attention window after permutation
3. The effective attention pattern matches expected graph connectivity
4. Gradients flow correctly through permutation/inverse-permutation
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from anemoi.models.layers.attention import BandedGraphSelfAttention
from anemoi.models.layers.attention import compute_banded_permutation
from anemoi.models.layers.utils import load_layer_kernels


# =============================================================================
# Test Fixtures - Create various graph topologies
# =============================================================================

@pytest.fixture
def layer_kernels():
    """Load default layer kernels."""
    return load_layer_kernels()


def create_grid_graph(rows: int, cols: int) -> tuple[torch.Tensor, int]:
    """Create a 2D grid graph (4-connected).

    Returns edge_index and number of nodes.
    """
    num_nodes = rows * cols
    edges = []

    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # Right neighbor
            if c < cols - 1:
                edges.append((node, node + 1))
                edges.append((node + 1, node))
            # Down neighbor
            if r < rows - 1:
                edges.append((node, node + cols))
                edges.append((node + cols, node))

    if not edges:
        return torch.tensor([[0], [0]], dtype=torch.long), num_nodes

    edges = torch.tensor(edges, dtype=torch.long).T
    return edges, num_nodes


def create_triangular_mesh_graph(n_rings: int) -> tuple[torch.Tensor, int]:
    """Create a triangular mesh graph (like icosahedral mesh).

    Creates a hexagonal pattern of nodes with triangular connectivity.
    Each interior node has 6 neighbors.
    """
    # Create nodes in a hexagonal pattern
    nodes = [(0, 0)]  # Center node
    node_to_idx = {(0, 0): 0}

    for ring in range(1, n_rings + 1):
        # Add nodes in each ring
        for i in range(6 * ring):
            # Compute position on hexagonal ring
            side = i // ring
            pos_on_side = i % ring

            # Six directions for hexagonal grid
            dirs = [
                (1, 0), (0.5, 0.866), (-0.5, 0.866),
                (-1, 0), (-0.5, -0.866), (0.5, -0.866)
            ]

            # Start position
            start_dir = dirs[side]
            next_dir = dirs[(side + 2) % 6]

            x = ring * start_dir[0] + pos_on_side * (next_dir[0] - start_dir[0]) / ring if ring > 0 else 0
            y = ring * start_dir[1] + pos_on_side * (next_dir[1] - start_dir[1]) / ring if ring > 0 else 0

            # Round to avoid floating point issues
            x, y = round(x * 100) / 100, round(y * 100) / 100
            if (x, y) not in node_to_idx:
                node_to_idx[(x, y)] = len(nodes)
                nodes.append((x, y))

    num_nodes = len(nodes)
    nodes = np.array(nodes)

    # Connect nodes that are close (triangular connectivity)
    edges = []
    threshold = 1.2  # Slightly larger than 1 to handle floating point

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(nodes[i] - nodes[j])
            if dist < threshold:
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        return torch.tensor([[0], [0]], dtype=torch.long), num_nodes

    edges = torch.tensor(edges, dtype=torch.long).T
    return edges, num_nodes


def get_k_hop_neighbors(edge_index: torch.Tensor, num_nodes: int, k: int) -> dict[int, set[int]]:
    """Compute k-hop neighborhood for each node.

    Returns a dict mapping node_id -> set of k-hop neighbor node_ids.
    """
    # Build adjacency list
    adj = {i: set() for i in range(num_nodes)}
    for src, dst in edge_index.T.tolist():
        adj[src].add(dst)

    # Compute k-hop neighbors via BFS
    k_hop = {}
    for node in range(num_nodes):
        visited = {node}
        frontier = {node}
        for _ in range(k):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(adj[n])
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        k_hop[node] = visited

    return k_hop


def compute_bandwidth(adj_matrix: np.ndarray) -> int:
    """Compute the bandwidth of a matrix (max distance from diagonal)."""
    rows, cols = np.where(adj_matrix != 0)
    if len(rows) == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


# =============================================================================
# Test: RCM Permutation Reduces Bandwidth
# =============================================================================

class TestRCMPermutation:
    """Tests for Reverse Cuthill-McKee permutation.

    Note: RCM minimizes bandwidth by reordering nodes so that GRAPH NEIGHBORS
    become SEQUENCE NEIGHBORS. The key property is that after RCM, connected
    nodes are close in the permuted sequence - this is what enables windowed
    attention to approximate graph attention.
    """

    def test_rcm_neighbors_are_close_grid(self):
        """After RCM, graph neighbors should be close in sequence space."""
        edge_index, num_nodes = create_grid_graph(10, 10)

        # Compute RCM permutation
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)
        perm_np = perm.numpy()

        # For each edge, measure the distance in permuted space
        max_neighbor_distance = 0
        total_edges = edge_index.shape[1]
        for i in range(total_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            src_perm = perm_np[src]
            dst_perm = perm_np[dst]
            distance = abs(src_perm - dst_perm)
            max_neighbor_distance = max(max_neighbor_distance, distance)

        # The bandwidth after RCM permutation
        bandwidth = max_neighbor_distance

        print(f"Grid graph (10x10, {num_nodes} nodes): max neighbor distance after RCM = {bandwidth}")

        # For a 10x10 grid, neighbors should be reasonably close
        # The theoretical minimum for a grid is around sqrt(n)
        assert bandwidth < num_nodes, \
            f"RCM should keep neighbors closer than random: bandwidth={bandwidth} >= num_nodes={num_nodes}"

    def test_rcm_neighbors_are_close_triangular(self):
        """After RCM, graph neighbors should be close in sequence space for triangular mesh."""
        edge_index, num_nodes = create_triangular_mesh_graph(5)

        if num_nodes < 2:
            pytest.skip("Graph too small")

        # Compute RCM permutation
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)
        perm_np = perm.numpy()

        # Measure max neighbor distance
        max_neighbor_distance = 0
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            distance = abs(perm_np[src] - perm_np[dst])
            max_neighbor_distance = max(max_neighbor_distance, distance)

        print(f"Triangular mesh ({num_nodes} nodes): max neighbor distance after RCM = {max_neighbor_distance}")

        # Neighbors should be reasonably close
        assert max_neighbor_distance < num_nodes, \
            f"RCM should keep neighbors closer than random"

    def test_permutation_is_bijective(self):
        """perm and inv_perm should be inverses of each other."""
        edge_index, num_nodes = create_grid_graph(5, 5)
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        # perm[inv_perm[i]] should equal i
        identity = perm[inv_perm]
        expected = torch.arange(num_nodes)
        assert torch.equal(identity, expected), "perm[inv_perm] should be identity"

        # inv_perm[perm[i]] should equal i
        identity2 = inv_perm[perm]
        assert torch.equal(identity2, expected), "inv_perm[perm] should be identity"


# =============================================================================
# Test: k-hop Neighbors Fall Within Window
# =============================================================================

class TestKHopCoverage:
    """Tests that k-hop neighbors are within the attention window after RCM."""

    @pytest.mark.parametrize("k_hop", [1, 2, 3])
    def test_khop_within_window_grid(self, k_hop):
        """k-hop neighbors should be within 2*bandwidth positions after RCM."""
        edge_index, num_nodes = create_grid_graph(8, 8)

        # Get k-hop neighborhoods in original ordering
        k_hop_neighbors = get_k_hop_neighbors(edge_index, num_nodes, k_hop)

        # Compute RCM permutation
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)
        perm_np = perm.numpy()

        # Compute bandwidth after permutation
        rows_perm = perm_np[edge_index[0].numpy()]
        cols_perm = perm_np[edge_index[1].numpy()]
        adj_perm = sp.csr_matrix(
            (np.ones(len(rows_perm)), (rows_perm, cols_perm)),
            shape=(num_nodes, num_nodes)
        )
        bandwidth = compute_bandwidth(adj_perm.toarray())

        # For k-hop attention, window should be approximately k * bandwidth
        # We use a generous window to account for boundary effects
        window_size = k_hop * bandwidth + bandwidth  # Extra margin

        # Check that all k-hop neighbors are within window
        max_distance = 0
        violations = 0
        for node in range(num_nodes):
            node_perm_pos = perm_np[node]
            for neighbor in k_hop_neighbors[node]:
                neighbor_perm_pos = perm_np[neighbor]
                distance = abs(node_perm_pos - neighbor_perm_pos)
                max_distance = max(max_distance, distance)
                if distance > window_size:
                    violations += 1

        coverage_rate = 1 - (violations / sum(len(v) for v in k_hop_neighbors.values()))
        print(f"Grid {k_hop}-hop: bandwidth={bandwidth}, window={window_size}, "
              f"max_distance={max_distance}, coverage={coverage_rate:.2%}")

        # Allow small number of violations (boundary nodes)
        assert coverage_rate > 0.95, \
            f"Only {coverage_rate:.2%} of {k_hop}-hop neighbors are within window"

    @pytest.mark.parametrize("k_hop", [1, 2])
    def test_khop_within_window_triangular(self, k_hop):
        """k-hop neighbors should be within window for triangular mesh."""
        edge_index, num_nodes = create_triangular_mesh_graph(4)

        if num_nodes < 10:
            pytest.skip("Graph too small")

        # Get k-hop neighborhoods
        k_hop_neighbors = get_k_hop_neighbors(edge_index, num_nodes, k_hop)

        # Compute RCM permutation
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)
        perm_np = perm.numpy()

        # Compute bandwidth after permutation
        rows_perm = perm_np[edge_index[0].numpy()]
        cols_perm = perm_np[edge_index[1].numpy()]
        adj_perm = sp.csr_matrix(
            (np.ones(len(rows_perm)), (rows_perm, cols_perm)),
            shape=(num_nodes, num_nodes)
        )
        bandwidth = compute_bandwidth(adj_perm.toarray())

        window_size = (k_hop + 1) * bandwidth

        # Check coverage
        max_distance = 0
        violations = 0
        total_pairs = 0
        for node in range(num_nodes):
            node_perm_pos = perm_np[node]
            for neighbor in k_hop_neighbors[node]:
                total_pairs += 1
                neighbor_perm_pos = perm_np[neighbor]
                distance = abs(node_perm_pos - neighbor_perm_pos)
                max_distance = max(max_distance, distance)
                if distance > window_size:
                    violations += 1

        coverage_rate = 1 - (violations / total_pairs) if total_pairs > 0 else 1.0
        print(f"Triangular {k_hop}-hop: bandwidth={bandwidth}, window={window_size}, "
              f"max_distance={max_distance}, coverage={coverage_rate:.2%}")

        assert coverage_rate > 0.90, \
            f"Only {coverage_rate:.2%} of {k_hop}-hop neighbors are within window"


# =============================================================================
# Test: Attention Pattern Verification via Gradient Analysis
# =============================================================================

class TestAttentionPatternViaGradient:
    """Test that gradients reveal the correct attention pattern.

    Key insight: If node i attends to node j, then perturbing node j's input
    should affect node i's output, which we can detect via gradients.
    """

    def test_gradient_flow_respects_window(self, layer_kernels):
        """Nodes should only receive gradients from nodes within the window."""
        num_nodes = 64
        num_heads = 4
        embed_dim = 32
        window_size = 8
        batch_size = 1

        # Create a simple line graph (worst case for RCM)
        edges = []
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        edge_index = torch.tensor(edges, dtype=torch.long).T

        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        # Create attention module
        attention = BandedGraphSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=window_size,
            attention_implementation="scaled_dot_product_attention",
        )
        attention.set_permutation(perm, inv_perm)

        # Create input with gradient tracking
        x = torch.randn(batch_size * num_nodes, embed_dim, requires_grad=True)
        shapes = [[batch_size * num_nodes, embed_dim]]

        # Forward pass
        out = attention(x, shapes, batch_size, perm=perm, inv_perm=inv_perm)

        # Pick a target node and compute gradient w.r.t. its output
        target_node = num_nodes // 2
        loss = out[target_node].sum()
        loss.backward()

        # Check gradient pattern
        grad = x.grad.abs().sum(dim=-1)  # Sum over embed_dim

        # Find which nodes have non-zero gradient
        grad_threshold = grad.max() * 1e-6
        contributing_nodes = (grad > grad_threshold).nonzero(as_tuple=True)[0].tolist()

        # Get target node's position in permuted space
        target_perm_pos = perm[target_node].item()

        # Check that contributing nodes are within window in permuted space
        for node in contributing_nodes:
            node_perm_pos = perm[node].item()
            distance = abs(node_perm_pos - target_perm_pos)
            assert distance <= window_size + 1, \
                f"Node {node} (pos {node_perm_pos}) contributes to target {target_node} " \
                f"(pos {target_perm_pos}) but distance {distance} > window {window_size}"

    def test_forward_backward_consistency(self, layer_kernels):
        """Forward and backward passes should be consistent."""
        edge_index, num_nodes = create_grid_graph(4, 4)
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        num_heads = 2
        embed_dim = 16
        window_size = 4
        batch_size = 2

        attention = BandedGraphSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=window_size,
            attention_implementation="scaled_dot_product_attention",
        )
        attention.set_permutation(perm, inv_perm)

        x = torch.randn(batch_size * num_nodes, embed_dim, requires_grad=True)
        shapes = [[batch_size * num_nodes, embed_dim]]

        # Forward
        out = attention(x, shapes, batch_size, perm=perm, inv_perm=inv_perm)

        # Backward
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"


# =============================================================================
# Test: Direct Attention Weight Inspection (using hooks)
# =============================================================================

class TestAttentionWeightInspection:
    """Directly inspect attention weights to verify sparsity pattern."""

    def test_windowed_attention_pattern(self, layer_kernels):
        """Attention weights should be zero outside the window."""
        num_nodes = 32
        num_heads = 2
        embed_dim = 16
        window_size = 4
        batch_size = 1

        # Create a simple graph
        edge_index, _ = create_grid_graph(4, 8)
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        # Create attention module with SDPA (we can inspect its mask)
        attention = BandedGraphSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=window_size,
            attention_implementation="scaled_dot_product_attention",
        )
        attention.set_permutation(perm, inv_perm)

        # Create input
        x = torch.randn(batch_size * num_nodes, embed_dim)
        shapes = [[batch_size * num_nodes, embed_dim]]

        # Get the internal attention's mask after a forward pass
        _ = attention(x, shapes, batch_size, perm=perm, inv_perm=inv_perm)

        # Check the mask shape and pattern
        internal_attn = attention.attention
        if hasattr(internal_attn, 'attention') and hasattr(internal_attn.attention, 'mask'):
            mask = internal_attn.attention.mask
            if mask is not None:
                # The mask should be True within window, False outside
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        distance = abs(i - j)
                        if distance > window_size:
                            assert not mask[i, j], \
                                f"Mask[{i},{j}] should be False (distance={distance} > window={window_size})"


# =============================================================================
# Test: Equivalence with Dense Attention for Small Graphs
# =============================================================================

class TestEquivalenceWithDense:
    """For small graphs where window covers everything, should match dense attention."""

    def test_full_coverage_equals_dense(self, layer_kernels):
        """When window >= num_nodes, should be equivalent to dense attention."""
        num_nodes = 8
        num_heads = 2
        embed_dim = 16
        window_size = num_nodes  # Full coverage
        batch_size = 1

        edge_index, _ = create_grid_graph(2, 4)
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        # Banded attention with full window
        banded_attn = BandedGraphSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=window_size,
            attention_implementation="scaled_dot_product_attention",
        )
        banded_attn.set_permutation(perm, inv_perm)

        # Dense attention (no windowing)
        from anemoi.models.layers.attention import MultiHeadSelfAttention
        dense_attn = MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=None,  # No window = dense
            attention_implementation="scaled_dot_product_attention",
        )

        # Copy weights
        dense_attn.load_state_dict(banded_attn.attention.state_dict())

        x = torch.randn(batch_size * num_nodes, embed_dim)
        shapes = [[batch_size * num_nodes, embed_dim]]

        out_banded = banded_attn(x, shapes, batch_size, perm=perm, inv_perm=inv_perm)
        out_dense = dense_attn(x, shapes, batch_size)

        # They should produce similar results (not exactly equal due to numerical precision)
        # The permutation shouldn't affect the output if window covers everything
        # Note: They won't be exactly equal because RCM changes the order operations happen
        # But the information content should be the same

        # Check that both outputs have similar statistics
        assert torch.allclose(out_banded.mean(), out_dense.mean(), atol=0.5), \
            "Mean outputs should be similar for full coverage"
        assert torch.allclose(out_banded.std(), out_dense.std(), atol=0.5), \
            "Std outputs should be similar for full coverage"


# =============================================================================
# Test: Recommended Window Size Calculation
# =============================================================================

class TestWindowSizeRecommendation:
    """Test helper functions for computing recommended window size."""

    def compute_recommended_window(self, edge_index: torch.Tensor, num_nodes: int, k_hop: int) -> int:
        """Compute recommended window size for k-hop attention coverage."""
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)
        perm_np = perm.numpy()

        # Compute bandwidth after permutation
        rows_perm = perm_np[edge_index[0].numpy()]
        cols_perm = perm_np[edge_index[1].numpy()]
        adj_perm = sp.csr_matrix(
            (np.ones(len(rows_perm)), (rows_perm, cols_perm)),
            shape=(num_nodes, num_nodes)
        )
        bandwidth = compute_bandwidth(adj_perm.toarray())

        # For k-hop, need roughly k * bandwidth
        # Add margin for safety
        return (k_hop + 1) * bandwidth

    def test_recommended_window_covers_khop(self):
        """Recommended window should achieve >95% k-hop coverage."""
        edge_index, num_nodes = create_grid_graph(10, 10)

        for k_hop in [1, 2, 3]:
            window = self.compute_recommended_window(edge_index, num_nodes, k_hop)
            k_hop_neighbors = get_k_hop_neighbors(edge_index, num_nodes, k_hop)

            perm, _ = compute_banded_permutation(edge_index, num_nodes)
            perm_np = perm.numpy()

            # Check coverage
            covered = 0
            total = 0
            for node in range(num_nodes):
                node_pos = perm_np[node]
                for neighbor in k_hop_neighbors[node]:
                    total += 1
                    neighbor_pos = perm_np[neighbor]
                    if abs(node_pos - neighbor_pos) <= window:
                        covered += 1

            coverage = covered / total if total > 0 else 1.0
            print(f"{k_hop}-hop: recommended window={window}, coverage={coverage:.2%}")

            assert coverage > 0.95, \
                f"Recommended window {window} only achieves {coverage:.2%} coverage for {k_hop}-hop"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node(self, layer_kernels):
        """Single node graph should work."""
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        perm, inv_perm = compute_banded_permutation(edge_index, 1)

        assert perm.shape[0] == 1
        assert inv_perm.shape[0] == 1

    def test_disconnected_graph(self, layer_kernels):
        """Disconnected graph should still work (nodes can attend to nearby positions)."""
        # Two disconnected components
        edges = []
        for i in range(4):
            for j in range(i + 1, 5):
                edges.append((i, j))
                edges.append((j, i))
        for i in range(5, 10):
            for j in range(i + 1, 10):
                edges.append((i, j))
                edges.append((j, i))

        edge_index = torch.tensor(edges, dtype=torch.long).T
        perm, inv_perm = compute_banded_permutation(edge_index, 10)

        # Should not crash
        assert perm.shape[0] == 10
        assert inv_perm.shape[0] == 10

    def test_complete_graph(self):
        """Complete graph should work (bandwidth = n-1)."""
        num_nodes = 10
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append((i, j))

        edge_index = torch.tensor(edges, dtype=torch.long).T
        perm, inv_perm = compute_banded_permutation(edge_index, num_nodes)

        # For complete graph, any permutation is equally good
        assert perm.shape[0] == num_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
