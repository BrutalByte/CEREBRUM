"""
Tests for Phase 23 — Distributed Spark GraphX DSCF.

Verifies the defensive initializations and mocked PySpark context behaviors.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# In environments without PySpark, core.dscf_spark.SparkSession is None
from core.dscf_spark import SparkDSCFEngine

def test_spark_missing_init():
    # If spark wasn't imported successfully, SparkDSCFEngine raises ImportError
    import core.dscf_spark as ds
    original_session = ds.SparkSession
    ds.SparkSession = None
    
    with pytest.raises(ImportError, match="pyspark is required"):
        SparkDSCFEngine()
        
    ds.SparkSession = original_session


def test_spark_init_with_mock():
    import core.dscf_spark as ds
    original_session = ds.SparkSession
    
    # Mocking SparkSession
    mock_session = MagicMock()
    ds.SparkSession = mock_session
    
    engine = SparkDSCFEngine(spark_session=mock_session)
    assert engine.spark == mock_session
    
    ds.SparkSession = original_session


def test_detect_offline_graphframes_missing():
    import core.dscf_spark as ds
    original_session = ds.SparkSession
    mock_session = MagicMock()
    ds.SparkSession = mock_session
    
    engine = SparkDSCFEngine(spark_session=mock_session)
    
    # Force graphframes to not exist
    with patch.dict('sys.modules', {'graphframes': None}):
        with pytest.raises(ImportError, match="graphframes is required"):
            engine.detect_offline(MagicMock(), MagicMock())
            
    ds.SparkSession = original_session
