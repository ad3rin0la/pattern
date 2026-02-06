"""
ToPE MCP Integration Layer
===========================
Adapters for structural biology and chemical property data retrieval
via Model Context Protocol servers.

Design principles:
1. Non-breaking: Wraps existing clients (MCSAClient, PDBClient, etc.)
2. Fallback: MCP failures gracefully degrade to existing HTTP clients
3. Caching: Respects existing cache_dir structure
4. Async-ready: Uses asyncio for parallel structure retrieval
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base MCP Adapter Protocol
# ─────────────────────────────────────────────────────────────────────────────


class MCPAdapter(ABC):
    """Base class for MCP server adapters in ToPE pipeline."""

    def __init__(
        self,
        cache_dir: Path,
        enable_mcp: bool = True,
        fallback_to_http: bool = True,
    ):
        """
        Parameters
        ----------
        cache_dir : Path
            Local cache directory for retrieved data
        enable_mcp : bool
            Whether to attempt MCP retrieval (disable for debugging)
        fallback_to_http : bool
            Whether to fall back to HTTP clients on MCP failure
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mcp = enable_mcp
        self.fallback_to_http = fallback_to_http

        # Statistics tracking
        self.stats = {
            "mcp_success": 0,
            "mcp_failure": 0,
            "cache_hits": 0,
            "http_fallback": 0,
        }

    @abstractmethod
    async def fetch(self, identifier: str, **kwargs) -> Optional[Any]:
        """Fetch data via MCP server.

        Parameters
        ----------
        identifier : str
            Primary identifier (PDB ID, UniProt accession, etc.)
        **kwargs
            Additional parameters for the MCP tool

        Returns
        -------
        Any or None
            Retrieved data, or None on failure
        """
        pass

    def _cache_path(self, identifier: str, suffix: str = ".json") -> Path:
        """Generate cache file path for an identifier."""
        return self.cache_dir / f"{identifier}{suffix}"

    def _load_cache(self, identifier: str, suffix: str = ".json") -> Optional[Dict]:
        """Load data from cache if available."""
        cache_path = self._cache_path(identifier, suffix)
        if cache_path.exists():
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit: {identifier}")
            with open(cache_path) as f:
                return json.load(f)
        return None

    def _save_cache(self, identifier: str, data: Dict, suffix: str = ".json"):
        """Save data to cache."""
        cache_path = self._cache_path(identifier, suffix)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Cached: {identifier}")

    def report_stats(self):
        """Log retrieval statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return
        logger.info(
            f"{self.__class__.__name__} stats: "
            f"MCP success={self.stats['mcp_success']}, "
            f"MCP failure={self.stats['mcp_failure']}, "
            f"cache hits={self.stats['cache_hits']}, "
            f"HTTP fallback={self.stats['http_fallback']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AlphaFold Structure Adapter
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AlphaFoldStructure:
    """AlphaFold predicted structure metadata."""

    uniprot_id: str
    pdb_path: Path
    plddt_scores: np.ndarray  # Per-residue confidence
    model_version: int
    organism: Optional[str] = None

    @property
    def mean_plddt(self) -> float:
        """Average pLDDT across all residues."""
        return float(np.mean(self.plddt_scores))

    @property
    def is_high_confidence(self) -> bool:
        """Check if structure meets high-confidence threshold (pLDDT > 70)."""
        return self.mean_plddt > 70.0


class AlphaFoldMCPAdapter(MCPAdapter):
    """
    Retrieve AlphaFold structures via MCP server.

    Use cases for ToPE:
    - Supplement PDB with predicted structures for enzymes lacking crystallography
    - Access full-length proteins (PDB often has only domains)
    - Get confidence scores for attribution analysis validation

    MCP Tool Call Example:
    ----------------------
    {
        "tool": "alphafold_search",
        "parameters": {
            "uniprot_id": "P00698",  # DHFR human
            "version": 4  # AlphaFold v4
        }
    }
    """

    def __init__(self, cache_dir: Path, pdb_fallback_client=None, **kwargs):
        super().__init__(cache_dir, **kwargs)
        self.pdb_fallback = pdb_fallback_client

    async def fetch(
        self,
        identifier: str,
        version: int = 4,
        extract_active_site: bool = False,
        catalytic_residues: Optional[List[Tuple[str, int]]] = None,
    ) -> Optional[AlphaFoldStructure]:
        """
        Fetch AlphaFold structure by UniProt ID.

        Parameters
        ----------
        identifier : str
            UniProt accession (e.g., 'P00698')
        version : int
            AlphaFold model version (2, 3, or 4)
        extract_active_site : bool
            If True, extract 8Å sphere around catalytic residues
        catalytic_residues : list of (chain, resid), optional
            Catalytic residue positions for active-site extraction

        Returns
        -------
        AlphaFoldStructure or None
        """
        # Check cache first
        cached = self._load_cache(identifier, suffix=f"_af{version}.json")
        if cached:
            return self._deserialize_structure(cached)

        if not self.enable_mcp:
            return await self._fallback_http(identifier, version)

        try:
            # MCP tool call (pseudo-code - actual implementation depends on MCP SDK)
            result = await self._call_mcp_tool(
                tool="alphafold_search",
                parameters={
                    "uniprot_id": identifier,
                    "version": version,
                },
            )

            if result is None:
                raise ValueError("MCP returned None")

            # Parse AlphaFold mmCIF format
            structure = self._parse_alphafold_result(result, identifier, version)

            # Extract active site if requested
            if extract_active_site and catalytic_residues:
                structure = self._extract_active_site(structure, catalytic_residues)

            # Cache result
            self._save_cache(
                identifier,
                self._serialize_structure(structure),
                suffix=f"_af{version}.json",
            )

            self.stats["mcp_success"] += 1
            logger.info(f"AlphaFold MCP: {identifier} (pLDDT={structure.mean_plddt:.1f})")
            return structure

        except Exception as e:
            logger.warning(f"AlphaFold MCP failed for {identifier}: {e}")
            self.stats["mcp_failure"] += 1

            if self.fallback_to_http:
                return await self._fallback_http(identifier, version)
            return None

    async def _call_mcp_tool(self, tool: str, parameters: Dict) -> Optional[Dict]:
        """
        Call MCP tool via SDK.

        This is a placeholder - actual implementation depends on:
        - MCP Python SDK availability
        - Server configuration (local vs remote)
        - Authentication requirements
        """
        # TODO: Implement actual MCP SDK calls
        # Example pseudo-code:
        # client = MCPClient(server_url="...")
        # response = await client.call_tool(tool, parameters)
        # return response.result
        raise NotImplementedError("MCP SDK integration pending")

    async def _fallback_http(
        self, identifier: str, version: int
    ) -> Optional[AlphaFoldStructure]:
        """Fallback to direct AlphaFold DB HTTP API."""
        if self.pdb_fallback is None:
            return None

        self.stats["http_fallback"] += 1
        logger.debug(f"AlphaFold HTTP fallback: {identifier}")

        # Download from AlphaFold DB
        url = f"https://alphafold.ebi.ac.uk/files/AF-{identifier}-F1-model_v{version}.cif"
        # Implementation delegated to existing PDBClient logic
        # ... (integrate with existing download methods)
        return None

    def _parse_alphafold_result(
        self, result: Dict, uniprot_id: str, version: int
    ) -> AlphaFoldStructure:
        """Parse AlphaFold mmCIF data and extract pLDDT scores."""
        # TODO: Parse mmCIF format using BioPython
        # Extract B-factor column (contains pLDDT scores)
        raise NotImplementedError("mmCIF parsing pending")

    def _serialize_structure(self, struct: AlphaFoldStructure) -> Dict:
        """Convert structure to JSON-serializable dict."""
        return {
            "uniprot_id": struct.uniprot_id,
            "pdb_path": str(struct.pdb_path),
            "plddt_scores": struct.plddt_scores.tolist(),
            "model_version": struct.model_version,
            "organism": struct.organism,
        }

    def _deserialize_structure(self, data: Dict) -> AlphaFoldStructure:
        """Reconstruct structure from cached JSON."""
        return AlphaFoldStructure(
            uniprot_id=data["uniprot_id"],
            pdb_path=Path(data["pdb_path"]),
            plddt_scores=np.array(data["plddt_scores"]),
            model_version=data["model_version"],
            organism=data.get("organism"),
        )

    def _extract_active_site(
        self,
        structure: AlphaFoldStructure,
        catalytic_residues: List[Tuple[str, int]],
        radius: float = 8.0,
    ) -> AlphaFoldStructure:
        """Extract 8Å active-site sphere from full structure."""
        # TODO: Integrate with existing ActiveSiteExtractor
        raise NotImplementedError("Active-site extraction pending")


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL Kinetics & Substrate Adapter
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChEMBLKineticsData:
    """Kinetic parameters from ChEMBL bioactivity assays."""

    ec_number: str
    substrate_chembl_id: str
    substrate_smiles: str
    kcat: Optional[float] = None  # s^-1
    km: Optional[float] = None  # µM
    kcat_km: Optional[float] = None  # M^-1 s^-1
    ki: Optional[float] = None  # µM (inhibition constant)
    assay_type: Optional[str] = None  # 'F' (functional), 'B' (binding)
    target_uniprot: Optional[str] = None

    @property
    def catalytic_efficiency(self) -> Optional[float]:
        """Compute kcat/Km if both available."""
        if self.kcat is not None and self.km is not None:
            return (self.kcat / self.km) * 1e6  # Convert to M^-1 s^-1
        return self.kcat_km


class ChEMBLMCPAdapter(MCPAdapter):
    """
    Retrieve kinetic parameters and substrate properties via ChEMBL MCP.

    Enhances existing BRENDA/SABIO-RK pipeline with:
    - Broader substrate coverage (ChEMBL: ~2.4M bioactivity records)
    - Direct SMILES retrieval for substrate chemical properties
    - Inhibitor data (Ki values) for selectivity modeling

    MCP Tool Calls:
    ---------------
    1. get_bioactivity: Query by target (EC number mapped to UniProt)
    2. compound_search: Get substrate chemical properties

    Integration point: KineticsAggregator.collect()
    """

    def __init__(
        self,
        cache_dir: Path,
        brenda_fallback_aggregator=None,
        **kwargs,
    ):
        super().__init__(cache_dir, **kwargs)
        self.brenda_fallback = brenda_fallback_aggregator

        # You already have ChEMBL MCP server connected!
        self.mcp_available = True

    async def fetch(self, identifier: str, **kwargs) -> Optional[Any]:
        """Fetch data by identifier - delegates to specific methods."""
        return await self.fetch_kinetics_by_ec(identifier, **kwargs)

    async def fetch_kinetics_by_ec(
        self,
        ec_number: str,
        param_types: Optional[List[str]] = None,
        min_pchembl: float = 6.0,  # ≥ 1 µM potency
    ) -> List[ChEMBLKineticsData]:
        """
        Fetch kinetic parameters for an enzyme by EC number.

        Parameters
        ----------
        ec_number : str
            Enzyme Commission number (e.g., '3.4.21.4' for trypsin)
        param_types : list of str
            Activity types to retrieve (IC50, Ki, Kd, EC50)
        min_pchembl : float
            Minimum pChEMBL value (higher = more potent)

        Returns
        -------
        list of ChEMBLKineticsData
        """
        if param_types is None:
            param_types = ["IC50", "Ki", "Kd"]

        cache_key = f"{ec_number}_chembl"
        cached = self._load_cache(cache_key)
        if cached:
            return [ChEMBLKineticsData(**d) for d in cached]

        if not self.enable_mcp:
            return await self._fallback_brenda(ec_number)

        try:
            # Step 1: Map EC number to UniProt target(s)
            # This requires EC → UniProt mapping (from UniProt MCP or cache)
            uniprot_ids = await self._map_ec_to_uniprot(ec_number)

            if not uniprot_ids:
                logger.warning(f"No UniProt mapping for EC {ec_number}")
                return await self._fallback_brenda(ec_number)

            # Step 2: Query ChEMBL bioactivity for each target
            all_kinetics = []
            for uniprot_id in uniprot_ids:
                target_chembl_id = await self._uniprot_to_chembl_target(uniprot_id)
                if target_chembl_id is None:
                    continue

                # MCP tool call: get_bioactivity
                bioactivities = await self._call_mcp_tool(
                    tool="ChEMBL:get_bioactivity",
                    parameters={
                        "target_chembl_id": target_chembl_id,
                        "min_pchembl": min_pchembl,
                        # Activity types mapped to kinetic params
                        # IC50, Ki → km (inhibition), EC50 → kcat (activation)
                    },
                )

                # Parse results
                for activity in bioactivities or []:
                    kinetics = await self._parse_bioactivity(
                        activity, ec_number, uniprot_id
                    )
                    if kinetics:
                        all_kinetics.append(kinetics)

            # Cache aggregated results
            self._save_cache(
                cache_key,
                [k.__dict__ for k in all_kinetics],
            )

            self.stats["mcp_success"] += 1
            logger.info(
                f"ChEMBL MCP: {ec_number} → {len(all_kinetics)} kinetic entries"
            )
            return all_kinetics

        except Exception as e:
            logger.warning(f"ChEMBL MCP failed for {ec_number}: {e}")
            self.stats["mcp_failure"] += 1

            if self.fallback_to_http:
                return await self._fallback_brenda(ec_number)
            return []

    async def fetch_substrate_properties(
        self,
        chembl_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve substrate chemical properties (SMILES, MW, LogP, etc.).

        Used for substrate encoding in Phase 3 selectivity prediction.

        Parameters
        ----------
        chembl_id : str
            ChEMBL compound identifier (e.g., 'CHEMBL25' for aspirin)

        Returns
        -------
        dict or None
            Chemical properties: smiles, mw, alogp, hba, hbd, psa, qed
        """
        cached = self._load_cache(chembl_id, suffix="_substrate.json")
        if cached:
            return cached

        try:
            # MCP tool call: compound_search
            result = await self._call_mcp_tool(
                tool="ChEMBL:compound_search",
                parameters={
                    "chembl_id": chembl_id,
                },
            )

            if result is None:
                return None

            # Extract relevant properties
            props = {
                "chembl_id": chembl_id,
                "smiles": result.get("smiles"),
                "molecular_weight": result.get("mw_freebase"),
                "alogp": result.get("alogp"),  # Lipophilicity
                "hbd": result.get("hbd"),  # H-bond donors
                "hba": result.get("hba"),  # H-bond acceptors
                "psa": result.get("psa"),  # Polar surface area
                "qed": result.get("qed_weighted"),  # Drug-likeness
                "num_ro5_violations": result.get("num_ro5_violations"),
            }

            self._save_cache(chembl_id, props, suffix="_substrate.json")
            self.stats["mcp_success"] += 1
            return props

        except Exception as e:
            logger.warning(f"ChEMBL substrate fetch failed for {chembl_id}: {e}")
            self.stats["mcp_failure"] += 1
            return None

    async def _call_mcp_tool(self, tool: str, parameters: Dict) -> Optional[Any]:
        """Call ChEMBL MCP tool - you have this server already connected!"""
        # TODO: Implement using MCP SDK
        # Since ChEMBL MCP is already available in your session, this is straightforward
        raise NotImplementedError("MCP SDK integration pending")

    async def _map_ec_to_uniprot(self, ec_number: str) -> List[str]:
        """Map EC number to UniProt accession(s) via UniProt MCP or cache."""
        # TODO: Implement EC → UniProt mapping
        # Option 1: UniProt MCP server (if available)
        # Option 2: Use existing M-CSA mappings
        # Option 3: Pre-built EC → UniProt lookup table
        raise NotImplementedError("EC → UniProt mapping pending")

    async def _uniprot_to_chembl_target(self, uniprot_id: str) -> Optional[str]:
        """Map UniProt accession to ChEMBL target ID."""
        # ChEMBL target_search tool supports this
        raise NotImplementedError("UniProt → ChEMBL mapping pending")

    async def _parse_bioactivity(
        self,
        activity: Dict,
        ec_number: str,
        uniprot_id: str,
    ) -> Optional[ChEMBLKineticsData]:
        """Parse ChEMBL bioactivity record into kinetics data."""
        # Map activity types to kinetic parameters
        # IC50, Ki, Kd → km-like (inhibition/binding)
        # EC50 → kcat-like (activation/catalysis)
        standard_type = activity.get("standard_type")
        value = activity.get("standard_value")
        units = activity.get("standard_units")

        if value is None or units not in ["nM", "uM"]:
            return None

        # Convert to µM
        if units == "nM":
            value_um = value / 1000.0
        else:
            value_um = value

        return ChEMBLKineticsData(
            ec_number=ec_number,
            substrate_chembl_id=activity.get("molecule_chembl_id"),
            substrate_smiles=activity.get("canonical_smiles"),
            km=value_um if standard_type in ["IC50", "Ki", "Kd"] else None,
            ki=value_um if standard_type == "Ki" else None,
            assay_type=activity.get("assay_type"),
            target_uniprot=uniprot_id,
        )

    async def _fallback_brenda(self, ec_number: str) -> List[ChEMBLKineticsData]:
        """Fallback to existing BRENDA/SABIO-RK pipeline."""
        if self.brenda_fallback is None:
            return []

        self.stats["http_fallback"] += 1
        logger.debug(f"BRENDA fallback: {ec_number}")

        # Use existing KineticsAggregator
        # ... (integrate with existing collect() method)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PubChem Substrate Feature Adapter
# ─────────────────────────────────────────────────────────────────────────────


class PubChemMCPAdapter(MCPAdapter):
    """
    Retrieve substrate molecular descriptors via PubChem.

    Complements ChEMBL for substrates not in ChEMBL database.
    Provides additional 2D/3D descriptors for substrate encoding.

    Integration: FeatureComputer for substrate-aware features (Phase 3)
    """

    async def fetch(self, identifier: str, **kwargs) -> Optional[Any]:
        """Fetch data by identifier."""
        return await self.fetch_descriptors(identifier)

    async def fetch_descriptors(
        self,
        smiles: str,
    ) -> Optional[Dict[str, float]]:
        """
        Compute molecular descriptors from SMILES.

        Returns
        -------
        dict or None
            Descriptors: mw, logp, tpsa, hbd, hba, rotatable_bonds, etc.
        """
        # TODO: Implement PubChem descriptor calculation
        # Can use RDKit locally instead of MCP for faster computation
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Integration with Existing Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class MCPEnhancedPipeline:
    """
    Drop-in replacement for CurationPipeline with MCP support.

    Usage
    -----
    from tope_mcp_adapters import MCPEnhancedPipeline

    pipeline = MCPEnhancedPipeline(
        config=config,
        enable_mcp=True,
        mcp_adapters={
            "alphafold": True,
            "chembl": True,
            "pubchem": False,  # Use RDKit locally instead
        }
    )
    records = await pipeline.run_async()
    """

    def __init__(
        self,
        config,
        enable_mcp: bool = True,
        mcp_adapters: Optional[Dict[str, bool]] = None,
    ):
        self.config = config
        self.enable_mcp = enable_mcp
        self.adapters = mcp_adapters or {}

        # Initialize MCP adapters
        if self.adapters.get("alphafold", True):
            self.alphafold = AlphaFoldMCPAdapter(
                cache_dir=config.data_root / "raw" / "alphafold",
                enable_mcp=enable_mcp,
            )

        if self.adapters.get("chembl", True):
            self.chembl = ChEMBLMCPAdapter(
                cache_dir=config.data_root / "raw" / "chembl",
                enable_mcp=enable_mcp,
            )

        # Keep existing clients for fallback
        # ... (integrate with CurationPipeline)

    async def run_async(self):
        """Async version of CurationPipeline.run() with MCP support."""
        # TODO: Convert CurationPipeline.run() to async
        # Use asyncio.gather() for parallel structure retrieval
        pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize adapters
    alphafold = AlphaFoldMCPAdapter(
        cache_dir=Path("./cache/alphafold"),
        enable_mcp=True,
    )

    chembl = ChEMBLMCPAdapter(
        cache_dir=Path("./cache/chembl"),
        enable_mcp=True,
    )

    # Fetch AlphaFold structure for DHFR
    # structure = await alphafold.fetch("P00698", version=4)

    # Fetch kinetics for trypsin (EC 3.4.21.4)
    # kinetics = await chembl.fetch_kinetics_by_ec("3.4.21.4")

    print("MCP adapters initialized successfully")
