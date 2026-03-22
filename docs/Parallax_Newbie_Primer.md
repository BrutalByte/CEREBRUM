# Parallax: A Newbie's Guide

## What it is, why it exists, and why it matters — in plain English

**For:** Anyone who has heard of AI and wants to understand something genuinely new about it.
**Not required:** A technical background of any kind.

*Bryan Alexander Buchorn — March 2026*

---

## Start here: the problem with AI that nobody talks about

You've probably used ChatGPT, Claude, or a similar AI tool. You asked it a question and it gave you a confident, well-written answer. Maybe it was right. Maybe it wasn't. Maybe you couldn't tell.

That last part — *you couldn't tell* — is the real problem.

When a human expert answers a question, you can ask: "How do you know?" They can point to a source. A study, a precedent, a procedure manual, an experiment. Their reasoning is traceable. You can follow it and check it.

When an AI gives you an answer, it genuinely cannot tell you how it knows. It isn't hiding the answer — it doesn't have one. Its knowledge lives inside billions of invisible numbers that were adjusted during training, and there is no way to look inside and trace a specific fact back to a specific source.

This occasionally leads to a phenomenon researchers call *hallucination*: the AI confidently tells you something that is simply not true. Not because it's malfunctioning. Because it generates plausible-sounding text, and plausible sometimes diverges from correct.

In a casual conversation, hallucination is annoying. In a medical diagnosis, a legal brief, or a financial report, it can cause real harm.

**Parallax is a direct attempt to solve this problem** — not by making LLMs better at not hallucinating, but by asking a different question entirely: *what if the AI didn't need to guess in the first place?*

---

## The library and the librarian

Imagine two ways to answer the question: "Is there any connection between aspirin and colorectal cancer?"

**Method 1: The all-knowing librarian**

You ask a librarian who has read every book in the building. She thinks for a moment and says: "Yes, I believe there is a connection." You ask how she knows. She says: "I've read so much, it's just... in my head somewhere. I'm quite confident."

She might be right. She's usually right. But she might also be misremembering something she read, or confusing it with something else. And you have no way to check.

This is roughly how an AI language model works. It has "read" an enormous amount of text. Its answers come from patterns in that reading, stored in a form that cannot be directly inspected.

**Method 2: The indexed card catalog**

You go to a card catalog. You look up "Aspirin." It links you to "COX-2 enzyme" with a note: "inhibits." You look up "COX-2 enzyme." It links to "colorectal cancer" with a note: "overexpressed in." You've just traced a two-step path through verified, indexed facts.

You didn't need to trust anyone's memory. Every step of the reasoning is a real card in the catalog that you can pull out and verify.

This is roughly how a Knowledge Graph works. Every fact is stored as an explicit connection. Every connection has a name (the relationship type). Nothing is inferred; everything is stated.

**The problem:** the card catalog is great for looking things up, but terrible at reasoning. It can show you individual connections but it can't automatically follow chains of them across a large collection to answer complex questions. It needs a human to do the walking.

**Parallax:** what if you could teach the card catalog to walk itself?

---

## What Parallax actually does

Parallax takes a Knowledge Graph — a database of interconnected facts — and gives it the ability to reason through it automatically, following chains of connections the way a skilled researcher would, but much faster and without any possibility of inventing a fact that isn't there.

Here's how it works in three plain steps.

### Step 1: Organize the neighborhood

Before Parallax can reason through a graph, it looks at the overall structure and finds the *natural neighborhoods* — groups of facts that are more connected to each other than to the rest of the graph.

Think of it like a city. A city wasn't designed in one day with neighborhoods pre-drawn on a map. Neighborhoods emerged organically because people who share something in common — work, culture, proximity — end up interacting more with each other than with people across town.

A medical knowledge graph naturally develops neighborhoods too. All the nodes about drug mechanisms cluster together. All the nodes about genetic markers cluster together. All the nodes about clinical symptoms cluster together. These clusters emerge from the structure of the data itself.

Parallax identifies these clusters using an algorithm called DSCF — Dual-Signal Community Fusion. The "dual signal" part means it looks at two things simultaneously when deciding where a fact belongs:

- **The local signal:** "What neighborhood are my immediate neighbors in right now?"
- **The global signal:** "What arrangement makes the overall structure of the whole graph most sensible?"

Most existing algorithms only look at one signal. DSCF looks at both. If both signals agree, the decision is made confidently. If they disagree, the system balances them — starting by trusting the local view and gradually shifting toward the global view as it processes more of the graph. This is borrowed from a technique in metallurgy called *simulated annealing*, where you start with high temperature (lots of local flexibility) and cool slowly toward a stable final state.

The inspiration for the "two signals that must agree" idea came from aviation. Aircraft navigation computers don't trust a single sensor. They run three sensors simultaneously and take the *middle value*, automatically rejecting any single sensor that's giving an outlier reading. Parallax does something similar: it requires consensus between signals before committing a node to a neighborhood. This "mid-level voting" is what lets it correct itself, the same way an aircraft corrects navigation errors.

These neighborhoods are important because they become the equivalent of *attention heads* — a concept from the AI systems that power modern language models. Each neighborhood specializes on a domain of knowledge, just like each attention head in a language model specializes on a type of relationship.

### Step 2: Walk the graph with focus

Once the neighborhoods are mapped, Parallax can answer questions by walking the graph intelligently.

Suppose you ask: "What did Marie Curie discover that is radioactive?"

Parallax starts at the "Marie Curie" node. It looks at every fact directly connected to Marie Curie and scores each one: how relevant does it seem? Is it in the same neighborhood? How close is it to where the answer is likely to be?

It keeps the most promising directions and follows them outward — one hop, two hops, three hops. At each step, it's scoring and pruning, like a hiking guide who keeps the ten most promising trails open but won't let you wander down every rabbit hole.

The score for each step combines several things:
- **How similar** the next node is to the current one (using embeddings — mathematical representations of meaning)
- **What neighborhood** the next node belongs to — same neighborhood gets more credit, different neighborhood gets less
- **What type of relationship** connects them — some relationship types are more informative than others
- **How deep** into the graph we already are — the deeper, the more we discount additional hops

After a few hops, Parallax has a set of complete paths. It ranks them by their total score and returns the best ones.

The answer for our question might look like:

> Marie Curie → *discovered* → Polonium → *exhibits* → Radioactivity

Score: 0.94. Two hops. One neighborhood transition.

Every edge in that path is a real, verified fact in the graph. Nothing was invented. Nothing was guessed. If you want to know *why* Parallax ranked this answer highest, you can look at the score at each step and understand exactly what it saw.

### Step 3: Optionally, translate into English

For most applications, you want a human-readable answer, not a graph path. Parallax can pass its verified reasoning path to a language model — which then does something much simpler and safer than reasoning: it translates the path into a sentence.

"Yes. Marie Curie discovered Polonium, which is radioactive."

The language model didn't reason here. It didn't guess. It just rendered a fact that Parallax had already verified into plain English. This dramatically reduces the chance of hallucination, because the language model is no longer responsible for the *logic* — only the *wording*.

---

## Why this matters in the real world

Here are some concrete scenarios where "you can verify every step" changes everything.

**A doctor asks about a drug interaction.**
A language model might say: "Drug A and Drug B may interact." It sounds confident. But where does that come from? Is it a known interaction or a pattern that looked like one? With Parallax, the answer comes with a path: Drug A → inhibits → Enzyme X → required by → Drug B's metabolism. You can look up every edge. You can cite every step. A physician can evaluate the chain rather than just trust the conclusion.

**A lawyer asks about regulatory precedent.**
Legal reasoning requires citation. An invented connection is malpractice. Parallax's answer isn't "this regulation might apply" — it's "this regulation applies to this entity type, which this company is, because of this case." Every edge is verifiable.

**A security analyst asks how an attacker could reach a critical system.**
Parallax returns a chain of real vulnerability relationships: Exposed Port → leads to → Unpatched Service → exploitable by → Privilege Escalation → grants access to → Database. No hops in that chain are invented. The analyst knows exactly what to fix.

**A factory engineer asks why two sensors keep alarming together.**
Parallax, operating on a live stream of sensor data, finds the path: Temperature Sensor A → co-activates → Pressure Sensor B → monitors → Cooling Loop → controlled by → Valve C. The connection is structural, visible, and traceable to real equipment relationships.

---

## What makes this new

Parallax didn't exist before. The pieces it uses — knowledge graphs, graph traversal, community detection algorithms, attention mechanisms — all existed. What's new is the specific combination and the specific insight that drives it:

> **The natural neighborhoods of a knowledge graph are structurally equivalent to the attention heads of a Transformer model.**

If that's true — and the early evidence suggests it is — then a knowledge graph can do the same kind of sophisticated, multi-signal reasoning that language models do, but using verified facts instead of probabilistic weights.

The research is testing this claim rigorously. If the claim holds up across multiple benchmark datasets and domains, it opens the door to AI reasoning that is:

- **Verifiable:** you can check every step
- **Free of hallucination:** by construction, not by luck
- **Free of training data:** the reasoning is structural, not learned
- **Cheap to run:** no expensive inference calls to a large model for every question
- **Live:** it can reason over data that is changing in real-time

That's a genuinely different kind of AI than what currently dominates the field.

---

## The one-sentence version

**Parallax teaches a database of interconnected facts to reason through itself, the way a skilled researcher follows a trail of evidence — without ever making something up.**

---

## Want to go deeper?

- **The Plain Language Guide** (`docs/Parallax_Plain_Language_Guide.md`) — covers the same territory with more technical detail, including the math behind the attention formula and a full breakdown of the research hypotheses being tested.
- **The Interactive Walkthrough** (`examples/Validation_Walkthrough.ipynb`) — a Jupyter Notebook you can run that shows the algorithm forming communities, scoring edges, and tracing reasoning paths step by step, with live visualizations.
- **The Studio** (`ui/studio.py`) — the browser-based interface where you can load your own data and ask questions directly, with no code required.
- **The White Paper** (`docs/Parallax_White_Paper.md`) — the formal research document with the full mathematical treatment, benchmark methodology, and experimental results.

---

*Questions? Contact the author: Bryan Alexander Buchorn — bryan.alexander@buchorn.com*

*Parallax is dual-licensed. Free for personal, academic, and non-profit use. Commercial use requires a separate license agreement.*
