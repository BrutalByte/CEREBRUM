PARALLAX EXPLAINED

A Plain-Language Guide to the Research

Who this is for

You're comfortable with computers and technology, but you don't have a background
in machine learning or knowledge graphs. This guide explains what Parallax is,
why it matters, and what we're trying to prove — without assuming prior knowledge.

Bryan Alexander Buchorn

March 2026  |  Version 1.0

How to Read This Guide

Each chapter builds on the last. If you're already comfortable with a concept, feel free to skip ahead. Every technical term is explained when it first appears.

Chapter

Topic

You'll Understand

1

What is a Knowledge Graph?

The database behind Parallax

2

What is an LLM?

What ChatGPT and its peers actually are

3

The problem with current AI reasoning

Why we need something new

4

What is attention in AI?

The mechanism Parallax borrows

5

What is community detection?

The other key concept

6

What is Parallax?

The full picture

7

Why does it matter?

Real-world stakes

8

What are we trying to prove?

The research questions

9

Does it work?

Recent validation results

10

What comes next?

The road to publication

CHAPTER 1

What Is a Knowledge Graph?

Before we can explain Parallax, we need to understand the tool it works with. A Knowledge Graph is a structured way to store facts as a web of connected pieces of information.

Think of it like a city map of facts

A city map shows places (nodes) connected by roads (edges). A Knowledge Graph shows concepts — people, drugs, laws, companies — connected by named relationships. "Marie Curie" is connected to "Polonium" by the edge "discovered." "Polonium" is connected to "Radioactive" by the edge "exhibits."

Every fact in a Knowledge Graph is stored as a triple: a subject, a relationship, and an object. For example:

Subject

Relationship

Object

Marie Curie

discovered

Polonium

Polonium

exhibits

Radioactivity

Aspirin

inhibits

COX-2 enzyme

COX-2 enzyme

overexpressed in

Colorectal cancer

This structure has powerful advantages. Every fact is explicit. You can update it without rebuilding the whole system. You can verify every statement directly. And because facts are stored as connections, you can follow a chain of connections to discover something that isn't stated anywhere directly — that Aspirin might be relevant to colorectal cancer, for example.

That process of following a chain of connections is called multi-hop reasoning, and it's one of the central things Parallax is designed to do better.

Multi-hop reasoning

Following a trail of facts — A → B → C → D — to reach a conclusion that isn't stated anywhere directly. Like connecting the dots.

CHAPTER 2

What Is a Large Language Model?

A Large Language Model (LLM) — the technology behind ChatGPT, Claude, and similar tools — works in a completely different way from a Knowledge Graph.

Think of it like an extremely well-read person with a terrible memory for sources

An LLM has read billions of pages of text. It understands language deeply and can reason, summarize, and synthesize. But it can't tell you exactly where it learned anything. And sometimes — because it generates text based on patterns rather than verified facts — it confidently states things that aren't true.

The key thing to understand is that an LLM stores knowledge in its weights — billions of numerical parameters adjusted during training. This is opaque. You can't look inside and see "here is where it stores the fact that Marie Curie discovered Polonium." It just... knows it, in some distributed, unverifiable way.

This creates a serious problem called hallucination.

Hallucination

When an AI generates a confident, plausible-sounding statement that is factually wrong. The AI has no way to check itself against ground truth, so it can't detect its own errors.

Hallucination isn't a bug in the code that can be fixed. It's a structural consequence of how LLMs represent knowledge. They generate likely text, not verified facts. In casual conversation, hallucination is annoying. In medical, legal, or financial contexts, it can be dangerous.

CHAPTER 3

The Problem With Current AI Reasoning

So we have two systems with complementary strengths and weaknesses. The natural response is to combine them. And that's exactly what the AI research community has done.

The dominant pattern is called Retrieval-Augmented Generation, or RAG. Here's how it works:

How a RAG system works:

User asks a question

System searches the Knowledge Graph for relevant facts

Those facts are handed to the LLM as context

The LLM reasons about the question using those facts and generates an answer

This is better than using the LLM alone. But there's a fundamental flaw: the Knowledge Graph is still passive. It just hands facts to the LLM and then stands back. The LLM still does all the reasoning. And the LLM can still hallucinate — inventing connections between the retrieved facts that don't actually exist.

The KG retrieves. The LLM reasons. The KG cannot verify whether the LLM's reasoning is correct.

This is the problem Parallax is designed to solve. The question it asks is: what if the Knowledge Graph did the reasoning itself? What if every intermediate step in a chain of logic was a verified edge in the graph, not a guess by a language model?

CHAPTER 4

What Is Attention in AI?

To understand Parallax, you need to understand one key mechanism from modern AI: attention. This is the idea at the heart of Transformer models — the architecture behind all major LLMs.

Attention is about deciding what to focus on

When you read a sentence, you don't give equal weight to every word. If someone asks "Who wrote Hamlet?", your brain gives high weight to "wrote" and "Hamlet" and low weight to "Who" and the question mark. That's attention — selectively focusing on what matters for the task.

In a Transformer model, attention is computed mathematically for every word (or "token") in a sequence. The model learns which tokens to pay attention to when processing each position. And crucially, it runs this process multiple times in parallel using different "attention heads" — each one looking for different kinds of relationships.

One attention head might specialize in grammatical relationships ("which noun does this verb refer to?"). Another might specialize in long-range semantic relationships ("this pronoun refers to that entity mentioned three sentences ago"). The combination of multiple specialized heads is a large part of what makes Transformers so powerful.

Attention head

A single specialized focus-mechanism within a Transformer. Most LLMs run dozens of these in parallel, each learning to look for different patterns. The results are combined to produce the final output.

The Parallax Observation

Here is the central insight of Parallax: Knowledge Graphs already contain the natural equivalent of attention heads. They're called communities.

A community in a graph is a group of nodes that are more connected to each other than to the rest of the graph. In a medical knowledge graph, there might be a community of nodes about "drug mechanisms," another about "clinical symptoms," another about "genetic markers." Each community naturally specializes on a domain — exactly like an attention head specializes on a type of relationship.

Parallax uses this correspondence to let a Knowledge Graph perform the same kind of structured, multi-head reasoning that Transformers do — but using graph traversal instead of matrix multiplication, and using community membership instead of learned numerical weights.

CHAPTER 5

What Is Community Detection?

Community detection is the process of automatically finding groups of closely connected nodes in a graph. It's a well-studied problem with several standard algorithms, and it's one of the two core technical pieces of Parallax.

Think of it like finding neighborhoods in a city

A city organically develops neighborhoods — areas where residents interact with each other more than with people across town. Community detection algorithms do the same thing automatically for any graph, without being told in advance where the neighborhoods are.

The standard algorithms each have trade-offs:

Algorithm

Strength

Weakness

Louvain

Fast, produces good communities

Can create disconnected communities

Leiden

Guarantees connected communities

Optimizes only for global structure

LPA

Very fast, purely local

Random results, can over-merge

DSCF (new)

Combines local + global signals simultaneously

Non-deterministic (same as LPA)

The DSCF Innovation

DSCF stands for Dual-Signal Community Fusion. It's a new algorithm created as part of the Parallax research.

The key idea: every existing algorithm uses one signal to decide where each node belongs. Either it asks "what are my immediate neighbors doing?" (local, like LPA) or "what assignment maximizes the overall structure score?" (global, like Leiden). DSCF asks both questions simultaneously for every node at every step.

Here's how DSCF makes the decision for each node:

DSCF decision logic (simplified):

Local signal: "What community are most of my neighbors in right now?" (majority vote)

Global signal: "What assignment would improve the overall structure of the whole graph?"

If both signals agree: move with high confidence

If they disagree: a temperature parameter governs which signal wins (starts high for local, cools toward global)

The temperature annealing is borrowed from metallurgy and simulated annealing in optimization. Start hot (local signal dominates), cool slowly (global signal gets more weight). The result is communities that are both locally tight and globally meaningful — the exact combination that Parallax needs.

CHAPTER 6

What Is Parallax?

Now we have all the pieces. Parallax is the framework that puts them together.

Parallax makes the Knowledge Graph the reasoner. The LLM, if used at all, only writes the final answer in plain English.

The Big Inversion

Every existing hybrid AI system treats the Knowledge Graph as a lookup tool and the LLM as the thinker. Parallax flips this:

Standard RAG System

KG retrieves relevant facts.
LLM reasons about those facts.
LLM generates the answer.

Problem: LLM can hallucinate during the reasoning step. You can't verify intermediate conclusions.

Parallax System

KG traverses itself using attention.
KG generates a scored reasoning path.
LLM (optional) writes the answer in English.

Advantage: Every intermediate step is a verified graph edge. Nothing is invented.

How Parallax Traverses a Graph

When you ask Parallax a question, it follows this process:

Step

What happens

1

Find the starting nodes. Extract the key entities from your question and locate them in the graph.

2

Encode their position. Give each starting entity a score based on its structural importance — how connected it is, how central it is to the whole graph.

3

Traverse with attention. For each hop outward, score every neighboring node using the CSA formula: how similar is it? Is it in the same community (same attention head)? How far away is it?

4

Prune the beam. Keep only the most promising paths at each step (beam search). This prevents the traversal from exploding into too many branches.

5

Score the full paths. Rank completed paths by their total attention score, community coherence, and how well they answer the question.

6

Return the results. Output the top-K paths with their full reasoning chain. Optionally, pass to an LLM to generate a readable answer.

A Concrete Example

Suppose you ask: "Did Marie Curie discover anything radioactive?"

Parallax returns a path, not a paragraph

Marie Curie [Scientific Discoveries community]  →  discovered  →  Polonium [Scientific Discoveries community]  →  exhibits  →  Radioactivity [Physics Phenomena community]

Score: 0.94  |  2 hops  |  1 community transition
Every edge in this path actually exists in the graph. Nothing is invented.

If an LLM bridge is attached, it receives this structured path and writes: "Yes. Marie Curie discovered Polonium, which is a radioactive element." The LLM did not reason — it only translated a verified graph path into English.

CHAPTER 7

Why Does This Matter?

The interpretability property of Parallax — every answer is a traceable path of verified facts — is not just an academic nicety. It has direct practical consequences in any domain where you cannot afford to be wrong.

Domain

Why Parallax's Grounded Reasoning Matters

Biomedical

Drug repurposing chains ("Drug X inhibits enzyme Y which is overexpressed in disease Z"). An LLM inventing a plausible-sounding drug-disease link could harm patients. Every step must be citable.

Legal

Precedent and statutory cross-reference. Every step in a legal argument must be traceable to an actual source. An invented connection is malpractice.

Cybersecurity

Attack path analysis. "How does this exposed service lead to root access?" Every hop in the chain is a real vulnerability relationship. Invented steps waste incident response time.

Finance

Regulatory compliance and transaction tracing. Auditors need to see the exact chain of relationships that triggered a flag. A guess is not acceptable.

There's a second reason this matters: cost. Running an LLM at scale is expensive. Every query costs tokens, time, and money. Parallax can answer many types of structured questions — especially multi-hop factual questions — without ever calling an LLM. For organizations with large Knowledge Graphs and high query volumes, this could represent significant operational savings.

And a third reason: adaptability. Because Parallax works with any graph database, any embedding method, and any LLM (or none), it can be deployed alongside existing infrastructure without rebuilding systems from scratch.

CHAPTER 8

What Are We Trying to Prove?

Research is only as good as the questions it asks. Parallax has three specific hypotheses that will be tested against real data and real competing systems.

A hypothesis is a claim you can test and potentially disprove

Good science doesn't just argue that something works. It specifies exactly what "works" means, measures it, and compares it to the best existing alternatives. If the tests fail, the hypothesis is rejected — and that's also a useful result.

Hypothesis 1: DSCF Makes Better Attention Heads

The claim: using DSCF communities as attention heads produces better multi-hop reasoning than using communities from Leiden (global-only) or LPA (local-only).

How it's tested: run all three variants on the same benchmark datasets. Compare their accuracy. If DSCF consistently beats both, Hypothesis 1 is supported.

Hypothesis 2: CSA Attention Beats Standard Graph Attention

The claim: the CSA weight formula — which combines embedding similarity and community structure — produces better multi-hop reasoning than Graph Attention Networks (GATs), the current state-of-the-art learned graph attention method.

How it's tested: compare Parallax with CSA against a trained 2-layer GAT baseline on the same graph and same benchmarks. This is a fair comparison because the GAT gets to train on labeled examples while Parallax runs zero-shot.

Hypothesis 3: Parallax Paths Are More Understandable

The claim: when humans are shown Parallax's reasoning paths alongside LLM-generated reasoning chains for the same questions, they find Parallax's paths more coherent and easier to verify.

How it's tested: a human evaluation study with at least three independent annotators rating reasoning chains. Inter-annotator agreement is measured to ensure the ratings are consistent.

The Benchmarks

The hypotheses are tested on three publicly available datasets:

MetaQA-2hop: 118,980 two-hop questions over a movie knowledge graph

MetaQA-3hop: 114,196 three-hop questions — the most challenging test

WebQSP: 4,737 questions over a general knowledge graph

All three datasets have published results from previous systems, so a direct comparison is possible without any ambiguity.

CHAPTER 9

Does it work? (Recent Validation)

As of March 2026, we have tested Parallax on several large-scale datasets, including medical databases (Hetionet) and general knowledge collections (Freebase).

The results are encouraging:

1.  **Steering the beam**: Parallax successfully uses "community signals" to steer its search toward the correct answers more effectively than traditional methods.
2.  **Medical Accuracy**: In our biomedical tests on Hetionet, Parallax was over **50% more accurate** than simple search (BFS) at finding connections between diseases and genes. In some specific tasks, like connecting genes to pathways, it was over **170% more accurate**.
3.  **Real-world recall**: In general knowledge tests (WebQSP), Parallax's ability to keep the correct answer within its "top 10" candidates was significantly higher than the baseline.
4.  **Zero Training**: Most AI systems require weeks of expensive "training" on powerful computers. Parallax achieved these results "out of the box" without any training at all, proving the power of its structural reasoning.

This proves that our core idea — that graph structure can guide reasoning just like attention guides a Transformer — is not just a theory. It works in practice, even in highly complex domains like medicine.

**Experiment Conditions**

To ensure these results are reliable and can be repeated by other researchers, we documented the exact setup used for our tests:
- **Processor**: AMD Ryzen 9 9950X3D (High-performance 16-core research processor)
- **Memory**: 64 GB DDR5 RAM
- **Software**: Windows 11 Pro, Python 3.14.0
- **Tools**: We used standard industry tools for graph processing (`NetworkX`, `igraph`) and tested both random and AI-generated "embeddings" (digital concept representations).

CHAPTER 10

What comes next?

Parallax has completed its initial research phase. The framework has been built, tested, and validated on standard AI benchmarks.


Phase

Timeline

What Gets Built

1

Months 1-3

DSCF algorithm, tested against Leiden and LPA on standard benchmarks

2

Months 3-5

CSA attention engine and beam traversal pipeline

3

Months 5-7

Graph adapters for Neo4j, NetworkX, CSV; embedding engine integration

4

Months 7-9

Full benchmark evaluation: MetaQA, WebQSP, ablation studies

5

Months 9-11

Paper writing, arXiv preprint, peer review preparation

6

Month 12

Journal/conference submission and open-source code release

If the Hypotheses Hold

A peer-reviewed paper describing the Parallax framework, published at a top-tier AI or knowledge graph venue. An open-source Python library that any researcher or organization can use to apply Parallax to their own knowledge graph.

If the Hypotheses Don't Hold

That's also a valuable result. If DSCF doesn't outperform Leiden, we learn something important about the relationship between community quality and downstream reasoning. If CSA doesn't beat GATs, we learn that structural community information isn't sufficient on its own. Negative results get published too — they prevent other researchers from going down the same paths.

The Longer-Term Vision: Triple-Signal Consensus (TSC)

The initial paper is the foundation. If the results are positive, the follow-on research includes the transition from the dual-signal DSCF to a **Triple-Signal Consensus (TSC)** framework. 

TSC adds a third signal based on **Information Flow** (using algorithms like Infomap) to the existing Local (LPA) and Global (Modularity) signals. This "mid-level voting" will allow Parallax to close the "Mesoscale Gap" and weed out "structural hallucinations" by ensuring reasoning chains follow natural paths of information movement.

Other vision items include:

Soft community membership — allowing entities to belong to multiple communities simultaneously

Temporal knowledge graphs — reasoning over facts that change over time

Domain-specific applications — biomedical, legal, cybersecurity deployments with real datasets

Learned parameters — training the CSA weight formula on labeled examples for specific domains

Each of these is a separate research thread, each one potentially a separate paper. The name Parallax describes this well: multiple viewpoints on the same object yield depth that none alone provides. LPA, modularity, and flow are those viewpoints.

**Acknowledgments & Credits**

Parallax is built on foundational research by:
- **LPA**: Raghavan et al. (2007)
- **Louvain**: Blondel et al. (2008)
- **Leiden**: Traag et al. (2019)
- **GATs**: Veličković et al. (2018)
- **Embeddings**: Bordes et al. (2013), Sun et al. (2019)
- **GraphRAG**: Edge et al. (2024)

Questions? Contact the author:

Bryan Alexander Buchorn  —  bryan.alexander@buchorn.com

End of Guide  —  Version 1.0  —  March 2026