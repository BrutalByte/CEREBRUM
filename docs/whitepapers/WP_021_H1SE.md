# WP_021: H1SE (Hop-1 Intermediate Seed Expansion)
## Overcoming Hub Dominance in Graph Search

**Status**: v2.51.0 (Phase 167 COMPLETE)

H1SE address the "Popularity Bias" in graph search. In graphs where a few nodes have very high degrees, traditional beam search is often captured by these hubs at the very first hop. H1SE solves this by giving every unique neighbor at hop 1 its own search budget, ensuring that specific, high-value reasoning paths are not crowded out by general-purpose hubs.
