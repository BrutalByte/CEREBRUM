
"""
CEREBRUM FastAPI REST server.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
import numpy as np
import json
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED
from core.attention_engine import CSAEngine, HomeostaticModulator

from api.schemas import (
    QueryRequest, QueryResponse, HealthResponse,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
_api_log = logging.getLogger("cerebrum.api")

_state: Dict[str, Any] = {
    "adapter": None,
    "community_map": None,
    "embeddings": None,
    "csa_metadata": None,
    "homeostatic_modulator": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _api_log.info("Lifespan: Cerebrum startup...")
    _state["homeostatic_modulator"] = HomeostaticModulator(target_activity=1.0)
    yield
    _api_log.info("Lifespan: Shutdown.")

app = FastAPI(
    title="CEREBRUM KG Reasoning API",
    description="Community-Structured Graph Attention Framework",
    version="2.33.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        adapter_loaded=_state["adapter"] is not None,
        communities_loaded=_state["community_map"] is not None,
        embeddings_loaded=_state["embeddings"] is not None,
        node_count=len(_state["embeddings"] or {}),
        community_count=len(set((_state["community_map"] or {}).values())),
    )

# ... (Additional routing would go here as identified in the API REST implementation)
