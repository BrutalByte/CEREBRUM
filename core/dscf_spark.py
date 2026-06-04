"""
Phase 23 â€” Distributed Spark GraphFrames DSCF (Offline).

Implements a Pregel-like message passing approach to compute
DSCF (Dual-Signal Community Fusion) signals across massive graphs.
"""
from typing import Counter, Optional
import os

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
except ImportError:
    SparkSession = None
    F = None

class SparkDSCFEngine:
    """
    Offline execution engine for massive-scale community detection.
    Maps the DSCF local update rules into Spark GraphFrames aggregate messages.
    """

    def __init__(self, spark_session: Optional['SparkSession'] = None):
        if not SparkSession:
            raise ImportError("pyspark is required for SparkDSCFEngine")
        self.spark = spark_session or self._get_default_session()
        
    def _get_default_session(self):
        try:
            return SparkSession.builder \
                .appName("CEREBRUM-SparkDSCF") \
                .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
                .getOrCreate()
        except Exception as e:
            raise RuntimeError(f"Failed to create Spark session: {e}")

    def detect_offline(self, vertices, edges, max_iter: int = 5, resolution: float = 1.0):
        """
        Runs a distributed DSCF iteration.
        
        vertices: PySpark DataFrame with at least 'id'
        edges: PySpark DataFrame with at least 'src', 'dst'
        """
        try:
            from graphframes import GraphFrame
            from graphframes.lib import AggregateMessages as AM
        except ImportError:
            raise ImportError("graphframes is required for SparkDSCFEngine")

        # Initialize communities to self id if missing
        if "community" not in vertices.columns:
            v_df = vertices.withColumn("community", F.col("id"))
        else:
            v_df = vertices
            
        # Ensure weight exists
        e_df = edges
        if "weight" not in e_df.columns:
            e_df = e_df.withColumn("weight", F.lit(1.0))
            
        g = GraphFrame(v_df, e_df)
        
        # Mocking the dual-signal phase: Label Propagation proxy
        # A true distributed Modularity pass requires global broadcasts of 2m and degree maps,
        # which is extremely expensive in MapReduce paradigms. 
        # This proxy distributes LPA votes and joins them.
        
        for p_iter in range(max_iter):
            # Send community ID to neighbors
            msg_to_dst = AM.src["community"]
            msg_to_src = AM.dst["community"]
            
            # Aggregate neighbors' communities
            agg = g.aggregateMessages(
                F.collect_list(AM.msg).alias("neighbor_communities"),
                sendToDst=msg_to_dst,
                sendToSrc=msg_to_src
            )
            
            # Determine majority vote locally 
            # (In PySpark >= 3.4 we can use array functions or a simple UDF)
            # UDF is easier for mock compatibility
            from collections import Counter
            def majority_vote(comms):
                if not comms: return None
                return Counter(comms).most_common(1)[0][0]
                
            vote_udf = F.udf(majority_vote, v_df.schema["id"].dataType)
            
            v_df = v_df.join(agg, on="id", how="left")
            v_df = v_df.withColumn("community", 
                             F.when(F.col("neighbor_communities").isNotNull(),
                                    vote_udf(F.col("neighbor_communities")))\
                              .otherwise(F.col("community")))
            v_df = v_df.drop("neighbor_communities")
            
            g = GraphFrame(v_df, e_df)
            
        return g.vertices
