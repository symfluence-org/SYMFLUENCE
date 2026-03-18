# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Graph-based operations for river network analysis.

Provides NetworkX graph construction and upstream basin tracing.
Eliminates code duplication across GeofabricDelineator and GeofabricSubsetter.

Supports multiple hydrofabric formats:
- MERIT: COMID with up1, up2, up3 columns
- TDX: streamID/LINKNO with USLINKNO1, USLINKNO2 columns
- NWS: divide_id with toid column (reverse direction)

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any, Dict, Set

import geopandas as gpd
import networkx as nx


class RiverGraphProcessor:
    """
    Graph operations for river network topology.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def build_river_graph(
        rivers: gpd.GeoDataFrame,
        fabric_config: Dict[str, Any]
    ) -> nx.DiGraph:
        """
        Build a directed graph representing the river network.

        The graph direction depends on the hydrofabric type:
        - MERIT/TDX: Edges point downstream (upstream → current)
        - NWS/TauDEM: Edges point upstream (current → downstream) if configured as 'downstream'

        Args:
            rivers: River network GeoDataFrame
            fabric_config: Configuration dict with keys:
                - 'river_id_col': Column name for river segment ID
                - 'upstream_cols': List of column names (upstream or downstream)
                - 'upstream_default': Default value indicating no link
                - 'direction': 'upstream' (default) or 'downstream'

        Returns:
            Directed graph of the river network
        """
        G = nx.DiGraph()

        # Determine flow direction handling
        # 'upstream': upstream_cols contain IDs of upstream segments (flow: upstream -> current)
        # 'downstream': upstream_cols contain IDs of downstream segments (flow: current -> downstream)
        direction = fabric_config.get('direction', 'upstream')

        # Auto-detect downstream-pointer columns (NWS toCOMID, HydroSHEDS NEXT_DOWN)
        downstream_cols = {'toCOMID', 'toid', 'NEXT_DOWN'}
        if fabric_config.get('upstream_cols') and set(fabric_config['upstream_cols']) <= downstream_cols:
            direction = 'downstream'

        for _, row in rivers.iterrows():
            current_basin = row[fabric_config['river_id_col']]
            G.add_node(current_basin)

            for up_col in fabric_config['upstream_cols']:
                linked_basin = row[up_col]

                # Skip if no link
                if linked_basin != fabric_config['upstream_default']:
                    if direction == 'downstream':
                        # Flow: current -> linked (downstream)
                        # We want the graph edges to represent flow direction?
                        # Usually river graphs are directed downstream.
                        # Wait, find_upstream_basins uses nx.ancestors.
                        # nx.ancestors(G, n) returns all nodes having a path to n.
                        # If edges are upstream -> downstream (A -> B), then ancestors of B includes A.
                        # So G should be directed A -> B (downstream).

                        # If direction is 'downstream' (current -> linked_basin),
                        # then we add edge (current, linked).
                        G.add_edge(current_basin, linked_basin)
                    else:
                        # If direction is 'upstream' (linked_basin -> current),
                        # then we add edge (linked, current).
                        G.add_edge(linked_basin, current_basin)

        return G

    @staticmethod
    def find_upstream_basins(
        basin_id: Any,
        G: nx.DiGraph,
        logger: Any
    ) -> Set:
        """
        Find all upstream basins for a given basin ID.

        Uses NetworkX ancestors to trace all basins upstream of the given basin.
        The result includes the basin itself.

        Args:
            basin_id: ID of the basin to find upstream basins for
            G: Directed graph of the river network
            logger: Logger instance for warnings

        Returns:
            Set of upstream basin IDs (including the given basin)
        """
        if G.has_node(basin_id):
            # Get all ancestors (upstream basins)
            upstream_basins = nx.ancestors(G, basin_id)
            # Include the basin itself
            upstream_basins.add(basin_id)
        else:
            logger.warning(f"Basin ID {basin_id} not found in the river network.")
            upstream_basins = set()

        return upstream_basins
