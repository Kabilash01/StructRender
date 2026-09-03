# StructRender — System Architecture & Knowledge Base

> **Status:** Final desired architecture — locked for implementation  
> **Project type:** Agent-authored, engine-grounded, neural game-rendering research system  
> **Current stage:** Early build / research prototype  
> **Primary research contribution:** Engine-Keyed Persistent Appearance Memory (EPAM)  
> **Primary renderer:** StructFormer — a lightweight hybrid CNN + window-transformer renderer  
> **Primary runtime target:** Godot + C++/GDExtension + TensorRT FP16 on an RTX 5060 Laptop GPU  
> **Training strategy:** Local debugging + free GPU experimentation + paid L40S final runs  
> **Project horizon:** Approximately 2.5 months  
> **Non-negotiable principle:** The deterministic engine owns world truth; the neural renderer owns appearance only.

---

## 1. Vision & Purpose

### The Problem

Interactive generative world models can synthesize convincing playable scenes directly from pixels and controller input, but they do not own authoritative game state. This creates structural and temporal failure modes that are unacceptable for a real game system:

1. **Spatial drift:** turn away and return, and geometry can shift.
2. **Logical drift:** collected items, NPC state, inventory, doors, or other gameplay facts can silently change.
3. **Appearance drift:** the same NPC, vehicle, prop, or door may be repainted differently after leaving and later re-entering the camera view.
4. **Appearance/logic entanglement:** visual restyling and world-state prediction are fused together instead of being separate concerns.

StructRender solves these problems by splitting the system into two authoritative domains:

- **Godot owns world truth** — physics, collision, transforms, entity identity, entity lifecycle, scoring, gameplay state, and simulation.
- **StructFormer owns appearance** — it converts deterministic structural buffers into styled RGB frames.
- **EPAM owns within-session generated appearance identity** — it remembers a compact appearance representation for a real engine entity and retrieves it when that same entity returns.

The project therefore asks one focused research question:

> **Can deterministic engine identity be used as an addressable visual-memory key so that a generative renderer preserves an object's appearance after the object leaves and later re-enters view?**

This is the headline contribution. Other mechanisms exist to support this experiment, not to compete with it for novelty.

### Why Now

StructRender is feasible because several capabilities are simultaneously practical:

- frontier coding agents can create and repair coherent multi-file Godot/C++/Python repositories;
- Godot provides deterministic physics, collision, scene state, camera data, render hooks, and entity lifecycle information;
- lightweight window-transformer image models can perform dense image-to-image transformation in one forward pass;
- high-quality diffusion models can be used **offline** as teachers without entering the runtime path;
- pretrained geometry-aware feature models can supervise structural correctness without training a 3D foundation model;
- ONNX and TensorRT provide a realistic path from PyTorch training to low-latency C++ inference;
- local and rented NVIDIA GPUs make a modest research-scale training budget practical.

The project deliberately avoids attempting to train a foundation-scale world model or custom coding LLM.

### Target Audience / Consumer

Primary consumers:

- the project author as a research and learning system;
- supervisors, professors, reviewers, and evaluators assessing the technical contribution;
- coding agents such as Codex, Claude Code, Cursor, or Devin operating on the repository;
- developers reproducing or extending the architecture;
- the Godot runtime consuming the exported inference engine.

There is no commercial SaaS or public-user product surface in the current scope.

---

## 2. Core Concept & Mental Model

### Central Metaphor

Think of StructRender as a theatre with four roles.

#### 1. Stage Manager — Godot

Godot knows every fact that must remain true:

- where every object is;
- which NPC exists;
- whether a coin was collected;
- whether a door is open;
- which entity spawned or was destroyed;
- whether a collision occurred;
- what the camera is doing.

Godot never asks the neural renderer to decide these facts.

#### 2. Set Painter — StructFormer

StructFormer receives deterministic scene structure and paints the final visible frame in the chosen style.

It does not control:

- physics;
- collision;
- entity identity;
- spawn/despawn;
- game rules.

#### 3. Wardrobe Ledger — EPAM

EPAM stores a compact generated-appearance memory for a real engine entity.

If `NPC_17` leaves view and returns 30 seconds later, EPAM provides the same identity-conditioned memory to the renderer.

The model does not visually guess whether the NPC is the same one. Godot tells it.

#### 4. Art Teacher — Offline Diffusion Teacher

A high-quality diffusion or other image-generation model may be used offline to create aligned stylized target images.

It is never required in the game loop.

### Design Philosophy

1. **Deterministic execution owns truth.**  
   Gameplay correctness never depends on generated pixels.

2. **Structure is never hallucinated when the engine already knows it.**  
   Depth, normals, motion, camera transforms, entity lifecycle, and identity come from Godot.

3. **Engine signals are used according to semantics.**  
   Motion vectors drive explicit warping. Entity IDs address masks and memory. They are not blindly concatenated as arbitrary image channels.

4. **EPAM is the protected novelty.**  
   If schedule or compute becomes constrained, optional visual-quality upgrades are removed before EPAM is removed.

5. **StructFormer is the main renderer.**  
   A CNN/U-Net remains the baseline. A one-step DiT is not required.

6. **Adversarial training is optional.**  
   A discriminator may improve sharpness, but the architecture is not defined as a GAN.

7. **Offline compute can be heavy; runtime must be lightweight.**  
   Diffusion teacher, geometry features, and expensive target generation may run in the cloud. Final inference must run locally in C++.

8. **Build-test-repair is mandatory for coding agents.**  
   Plausible code is not accepted until it is executed and validated.

9. **Baseline before novelty.**  
   Every new mechanism must be compared to a simpler working reference.

10. **Fail loudly offline; degrade gracefully online.**  
    Bad training data is rejected. Runtime renderer failure does not crash gameplay.

11. **Within-session identity only.**  
    EPAM does not claim cross-session persistence or zero-shot identity recovery.

12. **Reproducibility beats scale.**  
    A smaller, well-evaluated dataset and model are preferred over an expensive, poorly controlled experiment.

---

## 3. Architecture Overview

### System Topology

```text
                         USER GAME PROMPT
                                │
                                ▼
                    ┌─────────────────────┐
                    │ CODING AGENT SYSTEM │
                    │ Plan → Build → Run  │
                    │ Test → Repair       │
                    └─────────┬───────────┘
                              │
                              ▼
                       GENERATED GODOT GAME
                              │
                              ▼
                ┌──────────────────────────┐
                │ GODOT — SOURCE OF TRUTH │
                │ physics / logic / IDs   │
                └────────────┬─────────────┘
                             │
            ┌────────────────┼────────────────────┐
            │                │                    │
            ▼                ▼                    ▼
      STRUCTURAL DATA      MOTION DATA       ENTITY LIFECYCLE
      depth               motion vectors    spawn/destroy
      normals             camera matrices   visible/hidden
      material/albedo                       EntityKey
            │                │                    │
            └────────────────┼────────────────────┘
                             │
                             ▼
                      DATASET RECORDER
                             │
                             ▼
                      DATASET VALIDATOR
                             │
                             ▼
             NATIVE RGB + STYLE REFERENCE/PROMPT
                             │
                             ▼
                 OFFLINE DIFFUSION TEACHER
                             │
                             ▼
                   STYLIZED TARGET RGB
                             │
                             ▼
                    STRUCTFORMER TRAINING
                             │
               ┌─────────────┼─────────────┐
               │             │             │
               ▼             ▼             ▼
           structure      temporal       entity
           learning       learning       learning
                                             │
                                             ▼
                                            EPAM
                                             │
                                             ▼
                                      FINAL CHECKPOINT
                                             │
                                             ▼
                                            ONNX
                                             │
                ┌────────────────────────────┘
                │ copy to target laptop
                ▼
           TensorRT Builder
                │
                ▼
          FP16 TensorRT Engine
                │
                ▼
        C++ / GDExtension Runtime
                │
                ▼
                     LIVE GODOT LOOP
                │
                ├── capture structural buffers
                ├── warp previous features
                ├── build entity masks
                ├── lookup/update EPAM
                ├── run StructFormer
                └── display generated RGB
```

### Runtime Neural Topology

```text
                         CURRENT FRAME
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
      Depth                Normals          Material/Albedo
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                          CNN STEM
                             │
                             ▼
                  WINDOW TRANSFORMER ENCODER
                             │
             ┌───────────────┼───────────────────┐
             │               │                   │
             ▼               ▼                   ▼
      Current spatial   Temporal warped      Entity masks
         features          features               │
             │               │                    ▼
             │               │                EntityKey
             │               │                    │
             │               │                    ▼
             │               │                   EPAM
             │               │                    │
             │               │              Entity tokens
             │               │                    │
             └───────────────┼────────────────────┘
                             │
                             ▼
                 ENTITY-MASKED CROSS-ATTENTION
                             ▲
                             │
                      cached style token
                             │
                             ▼
                 WINDOW TRANSFORMER DECODER
                             │
                             ▼
                          CNN HEAD
                             │
                             ▼
                         STYLED RGB
```

### Execution Loop / Workflow

#### A. Agent Authorship — Offline

1. User supplies a game specification.
2. Coding agent generates an implementation plan.
3. Agent creates Godot scenes, scripts, collision, logic, entity lifecycle hooks, and instrumentation hooks.
4. Automated launch and scene-load checks run.
5. Tests verify:
   - scenes open;
   - gameplay logic executes;
   - entities expose stable identity;
   - instrumentation produces required buffers.
6. On failure:
   - collect logs;
   - diagnose root cause;
   - patch;
   - rerun.
7. The game is accepted only after required tests pass.

#### B. Data Collection — Offline

For each selected frame:

1. Godot advances deterministic simulation.
2. Recorder captures:
   - native RGB;
   - depth;
   - normals;
   - material/albedo representation;
   - motion vectors;
   - entity-ID map;
   - camera view/projection matrices;
   - entity lifecycle state;
   - style ID/prompt;
   - session ID and frame ID.
3. Dense contiguous windows are preserved for temporal training.
4. Dedicated disappearance/reappearance sequences are recorded for EPAM training and evaluation.
5. The validator checks all channels and sequence semantics.

#### C. Teacher Target Generation — Offline

1. Native RGB and style input are passed to an offline teacher.
2. Teacher produces high-quality styled target RGB.
3. Target and structural frame must remain frame-aligned.
4. Teacher is not part of runtime.
5. Teacher outputs are cached because regeneration is expensive.

#### D. StructFormer Training — Offline

Training follows a curriculum:

1. structure reconstruction;
2. geometry-aware supervision;
3. temporal consistency;
4. structured entity awareness;
5. EPAM;
6. final style refinement;
7. optional adversarial sharpening.

Each stage resumes from the previous best checkpoint.

#### E. Model Export — Offline

1. Select best checkpoint.
2. Export PyTorch model to ONNX.
3. Validate ONNX output numerically against PyTorch.
4. Transfer ONNX model to target RTX 5060 laptop.
5. Build TensorRT engine **on the target machine**.
6. Benchmark standalone inference before integrating with Godot.

#### F. Live Inference — Runtime

For every game frame:

1. Godot advances world state.
2. Structural buffers and entity lifecycle are produced.
3. Previous features are motion-warped.
4. Invalid/disoccluded history is masked.
5. Visible entity masks are built.
6. EPAM entries are looked up.
7. Cached style embedding is supplied.
8. TensorRT StructFormer generates RGB.
9. EPAM updates only from high-confidence observations.
10. Generated RGB is displayed.
11. On failure, native gameplay continues.

### Base Engine / Primary Rationale

#### Godot

Chosen because it provides:

- deterministic physics and collision;
- scene graph and scripting;
- open-source rendering stack;
- C++ GDExtension integration;
- suitable render instrumentation;
- no need to write a physics engine.

Rejected:

- **Custom engine:** too much non-research work and weakens the deterministic-ground-truth claim.
- **Unity/Unreal:** feasible but increases tooling/integration/licensing complexity without strengthening the core research question.

#### StructFormer

StructFormer is a **hybrid dense image-to-image transformer**, not a generic large ViT.

It combines:

- convolutional input stem;
- hierarchical local/window attention;
- entity-masked cross-attention;
- lightweight transformer decoder;
- convolutional output head.

Chosen because:

- one forward pass;
- more expressive global/object-level relationships than a pure CNN;
- naturally supports EPAM entity tokens and cross-attention;
- much lower deployment risk than a custom one-step diffusion transformer;
- can be designed specifically for ONNX/TensorRT.

Rejected as final renderer:

- **Pure CNN/U-Net:** retained as baseline, but weaker mechanism for tokenized entity memory and long-range interactions.
- **Classic GAN as architecture:** adversarial loss may be used, but a GAN discriminator is not a runtime model.
- **One-step DiT:** strong quality potential but too risky as a required deliverable under the project schedule, inference budget, and 8 GB local VRAM.
- **Large global ViT:** quadratic attention and poor real-time efficiency at image resolution.
- **Temporal foundation model:** unnecessary because the engine already provides deterministic motion and identity.

---

## 4. Components & Module Boundaries

### Agent Authorship Module

- **Ownership:**  
  Converts a natural-language game specification into a playable and instrumented Godot project.

- **Inputs & Outputs:**  
  Input:
  ```json
  {
    "game_spec": "Create a medieval arena game with enemies and collectible coins.",
    "constraints": {
      "engine": "Godot",
      "requires_entity_lifecycle": true,
      "requires_render_instrumentation": true
    }
  }
  ```

  Output:
  ```text
  game_project/
    project.godot
    scenes/
    scripts/
    instrumentation/
    tests/
  ```

- **Explicit Boundaries:**  
  - does **not** determine final visual style;
  - does **not** train StructFormer;
  - does **not** own runtime truth;
  - does **not** invent a second identity system;
  - does **not** accept untested generated code.

---

### Game Runtime Module

- **Ownership:**  
  Owns authoritative gameplay state.

- **Inputs & Outputs:**  
  Inputs:
  - controller/player input;
  - Godot scenes and scripts.

  Outputs:
  - physics state;
  - transforms;
  - entity lifecycle events;
  - entity identity;
  - camera state;
  - native RGB fallback;
  - structural rendering inputs.

- **Explicit Boundaries:**  
  - never reads generated RGB to make gameplay decisions;
  - never receives state corrections from StructFormer;
  - never lets EPAM determine whether an entity exists;
  - never treats a neural feature as authoritative game state.

---

### Structural Export Module

- **Ownership:**  
  Produces the deterministic per-frame rendering structure needed for training and inference.

- **Inputs & Outputs:**  
  Output schema:
  ```python
  FrameStructure = {
      "depth": Tensor[H, W, 1],
      "normal": Tensor[H, W, 3],
      "albedo_or_material": Tensor[H, W, C],
      "motion": Tensor[H, W, 2],
      "entity_id_map": IntTensor[H, W],
      "camera_view": Tensor[4, 4],
      "camera_projection": Tensor[4, 4]
  }
  ```

- **Explicit Boundaries:**  
  - does not fill missing buffers with zeros silently;
  - does not convert categorical entity IDs into continuous-valued semantic features;
  - does not estimate motion from pixels if deterministic engine motion is available.

---

### Entity Lifecycle Adapter

- **Ownership:**  
  Converts engine spawn/despawn/visibility events into stable StructRender lifecycle events.

- **Inputs & Outputs:**  
  Example:
  ```python
  EntityEvent = {
      "session_id": "session_004",
      "instance_id": 17,
      "generation": 0,
      "event": "spawn",
      "frame_id": 842
  }
  ```

- **Explicit Boundaries:**  
  - does not infer entity identity visually;
  - does not reuse EPAM entries after destruction;
  - does not assume low-level engine indices are never recycled.

---

### EntityKey Service

- **Ownership:**  
  Generates the stable session-unique address used by EPAM.

- **Canonical Format:**
  ```text
  <session_id>:<instance_id>:<generation>
  ```

  Example:
  ```text
  session_004:npc_17:0
  ```

- **Explicit Boundaries:**  
  - numeric ID magnitude has no semantic meaning;
  - keys are not learned embeddings;
  - new entities never inherit old appearance memory simply because an internal index was reused.

---

### Dataset Recorder

- **Ownership:**  
  Writes synchronized frame, target, and entity data to persistent storage.

- **Inputs & Outputs:**  
  Input:
  - Godot structural output;
  - native RGB;
  - entity events;
  - style metadata.

  Output:
  - tensor/image files;
  - JSONL/Parquet manifest;
  - dataset sequence metadata.

- **Explicit Boundaries:**  
  - does not repair corrupted data;
  - does not train models;
  - does not sample away all temporal continuity;
  - does not overwrite validated immutable datasets.

---

### Dataset Validator

- **Ownership:**  
  Rejects invalid training sessions and catches structural misalignment.

- **Checks:**  
  - shape consistency;
  - finite depth;
  - valid normals;
  - motion dimensions;
  - entity map integer type;
  - frame ordering;
  - target availability;
  - entity lifecycle consistency;
  - camera matrix validity.

- **Explicit Boundaries:**  
  - does not silently "fix" tensors;
  - does not infer missing frames;
  - does not accept invalid data because a training run has already started.

---

### Offline Teacher Target Generator

- **Ownership:**  
  Generates high-quality styled target frames for supervised training.

- **Inputs & Outputs:**  
  Inputs:
  - native engine RGB;
  - style prompt or reference.

  Outputs:
  - frame-aligned styled RGB.

- **Explicit Boundaries:**  
  - never runs inside the game loop;
  - never owns structure or entity identity;
  - never becomes a runtime dependency;
  - does not need to fit inside the local 8 GB deployment budget.

---

### StructFormer CNN Stem

- **Ownership:**  
  Converts continuous structural inputs into efficient local feature maps.

- **Inputs & Outputs:**  
  Inputs:
  - normalized depth;
  - normals;
  - material/albedo features.

  Output:
  - spatial feature map.

- **Rationale:**  
  Convolution is computationally efficient for local edges, depth discontinuities, and normal boundaries.

- **Explicit Boundaries:**  
  - raw entity IDs are not concatenated here;
  - motion is not treated as an unexplained appearance channel.

---

### Window Transformer Encoder

- **Ownership:**  
  Models spatial relationships with bounded local attention and hierarchical receptive fields.

- **Inputs & Outputs:**  
  Input:
  - CNN stem features.

  Output:
  - transformer spatial features.

- **Rationale:**  
  Local/window attention avoids the cost of full-resolution global self-attention while allowing broader object-level interaction than a shallow CNN.

- **Explicit Boundaries:**  
  - no unbounded global attention at full resolution;
  - no massive ViT-scale model;
  - no temporal history prediction.

---

### Engine-Guided Temporal Reprojection Module

- **Ownership:**  
  Aligns previous-frame renderer features to current-frame coordinates using deterministic motion.

- **Core Operation:**
  \[
  \tilde{F}_{t-1}=W(F_{t-1},V_t)
  \]

  where:
  - \(F_{t-1}\): previous feature map;
  - \(V_t\): engine motion field;
  - \(W\): warp/reprojection operation.

- **Confidence Mask:**
  \[
  e_D = |D_t - W(D_{t-1},V_t)|
  \]

  \[
  e_N = 1 - N_t \cdot W(N_{t-1},V_t)
  \]

  \[
  M_t = \exp(-\alpha e_D-\beta e_N)
  \]

- **Explicit Boundaries:**  
  - production path does not depend on RAFT;
  - invalid warped history is discarded;
  - newly disoccluded pixels are not filled from stale history;
  - temporal fusion never overrides current engine structure.

---

### Entity Mask Builder

- **Ownership:**  
  Converts the categorical entity-ID map into masks.

- **Example:**
  \[
  S_i(x,y)=\mathbb{1}[ID(x,y)=i]
  \]

- **Inputs & Outputs:**  
  Input:
  ```python
  entity_id_map: IntTensor[H, W]
  ```

  Output:
  ```python
  {
      EntityKey: {
          "mask": BoolTensor[H, W],
          "pixel_count": int,
          "visibility": float
      }
  }
  ```

- **Explicit Boundaries:**  
  - does not interpret `ID=10` as more similar to `ID=11` than `ID=900`;
  - does not replace deterministic identity with segmentation or visual re-identification.

---

### EPAM — Engine-Keyed Persistent Appearance Memory

- **Ownership:**  
  Preserves generated appearance identity for previously observed live entities during the current session.

- **Memory Entry:**
  ```python
  MemoryEntry = {
      "embedding": Tensor[D_memory],
      "first_seen_frame": int,
      "last_seen_frame": int,
      "last_update_frame": int,
      "confidence": float,
      "style_id": str
  }
  ```

- **Update Rule:**
  \[
  M_i^t =
  \gamma M_i^{t-1}
  +(1-\gamma)F_i^t
  \]

  only if:
  \[
  C_i^t>\tau
  \]

- **Lifecycle:**
  ```text
  SPAWN
    ↓
  first reliable observation
    ↓
  CREATE
    ↓
  high-confidence visible observation
    ↓
  UPDATE
    ↓
  leaves view
    ↓
  KEEP UNCHANGED
    ↓
  reappears
    ↓
  LOOKUP SAME MEMORY
    ↓
  entity destroyed
    ↓
  DELETE
  ```

- **Style Rule:**  
  v1 clears EPAM on style change.

- **Capacity Rule:**  
  Core experiment keeps memory for every live, previously seen entity. No arbitrary LRU eviction is used in the primary benchmark.

- **Explicit Boundaries:**  
  - no cross-session persistence;
  - no memory after destroy;
  - no memory reuse after ID recycling;
  - no zero-shot appearance reconstruction for unseen entities;
  - no semantic claim that appearance embedding is a complete object description;
  - no stale cross-style memory in v1.

---

### Entity Feature Extractor

- **Ownership:**  
  Pools spatial features belonging to a deterministic entity mask.

- **Core Operation:**
  \[
  F_i=\operatorname{Pool}(F\odot S_i)
  \]

- **Explicit Boundaries:**  
  - does not choose entity identity;
  - does not create masks from generated RGB;
  - does not update EPAM under low confidence.

---

### EPAM Entity Token Projector

- **Ownership:**  
  Converts an EPAM appearance embedding into a transformer-compatible entity token.

- **Inputs & Outputs:**  
  Input:
  ```python
  embedding: Tensor[D_memory]
  ```

  Output:
  ```python
  entity_token: Tensor[D_model]
  ```

- **Explicit Boundaries:**  
  - does not embed raw entity numbers;
  - does not create a learned identity system disconnected from engine IDs.

---

### Entity Token Packing & Overflow Policy

- **Ownership:**  
  Selects which visible entities receive an EPAM token slot on frames where the number of visible entities exceeds `K_max`.

- **Context:**  
  TensorRT requires fixed-shape tensors, so the number of entity token slots consumed per frame must be bounded at `K_max`. Without a defined selection rule, an implicit/arbitrary drop order (e.g. spawn order or engine iteration order) risks silently dropping the exact entity a reappearance test depends on.

- **Selection Rule:**  
  Rank currently visible entities by `pixel_count` (from the Entity Mask Builder output), descending, and keep the top `K_max`. Ties broken by lower `EntityKey` instance_id for determinism.

  \[
  \text{kept} = \operatorname{top\_k}\big(\{S_i : i \in \text{visible}\},\ K_max,\ \text{key}=\text{pixel\_count}\big)
  \]

- **Dropped-Entity Behavior:**  
  An entity excluded on a given frame by this rule:
  - does not receive an entity token that frame, and is rendered using structural/temporal features only (no EPAM conditioning that frame);
  - is **not** deleted from EPAM and is **not** treated as having left view for lifecycle/confidence-gating purposes;
  - becomes eligible again the moment its `pixel_count` re-enters the top `K_max`.

- **Evaluation Implication:**  
  The core reappearance benchmark (Milestone 8/9) is run with entity counts kept at or below `K_max` so headline RCS results are not confounded by overflow drops. Overflow behavior under crowded scenes (more visible entities than `K_max`) is reported separately as a secondary robustness result, not folded into the headline figure.

- **Explicit Boundaries:**  
  - does not use spawn order, engine iteration order, or arbitrary/random drop order;
  - does not silently vary `K_max` per frame;
  - does not treat a dropped-that-frame entity as destroyed or as having exited view.

---

### Entity-Masked Cross-Attention Module

- **Ownership:**  
  Allows spatial tokens to consume the correct entity's EPAM token.

- **Attention Rule:**
  \[
  A_{p,i}=
  \begin{cases}
  Q_pK_i^T/\sqrt d,&p\in S_i\\
  -\infty,&p\notin S_i
  \end{cases}
  \]

  The engine mask determines legal attention connections.

- **Rationale:**  
  An NPC's spatial features must not retrieve a car's appearance memory.

- **Explicit Boundaries:**  
  - no unrestricted entity-token mixing;
  - no memory addressing based on visual similarity;
  - no categorical IDs as continuous image values.

---

### Style Conditioning Module

- **Ownership:**  
  Produces and caches a style representation.

- **Workflow:**
  ```text
  style prompt/reference
        ↓
  style encoder
        ↓
  style token
        ↓
  cache until style changes
  ```

- **Explicit Boundaries:**  
  - does not run a heavy text encoder every frame;
  - changing style invalidates v1 EPAM;
  - style cannot mutate game state.

---

### Window Transformer Decoder

- **Ownership:**  
  Reconstructs high-resolution features after structure, temporal, style, and entity-memory fusion.

- **Explicit Boundaries:**  
  - does not predict future frames;
  - does not own memory lifecycle;
  - does not perform entity identification.

---

### CNN RGB Head

- **Ownership:**  
  Converts final decoder features into RGB.

- **Output:**
  ```python
  rgb: Tensor[B, 3, H, W]
  ```

- **Explicit Boundaries:**  
  - no game-state output;
  - no segmentation output required in the runtime path.

---

### Training Loss Module

- **Ownership:**  
  Defines supervised and consistency objectives.

Core final objective:

\[
L =
\lambda_{rec}L_{rec}
+\lambda_{perc}L_{perc}
+\lambda_{geo}L_{geo}
+\lambda_{temp}L_{temp}
+\lambda_{entity}L_{entity}
+\lambda_{adv}L_{adv}^{optional}
\]

#### Reconstruction

\[
L_{rec}=||I_t-\hat I_t||_1
\]

#### Perceptual

Feature-space similarity to improve visual quality.

#### Geometry

\[
L_{geo}=d(F_{3D}(I_t),F_{3D}(\hat I_t))
\]

using a frozen geometry-aware feature model.

#### Temporal

\[
L_{temp}
=
M_t\odot||I_t-W(I_{t-1},V_t)||_1
\]

#### Entity Consistency

\[
L_{entity}
=
1-\cos(F_i^t,F_i^{t+k})
\]

with special importance on large \(k\) after the entity leaves view.

#### Optional Adversarial

A small PatchGAN-style discriminator may be introduced only if non-adversarial training produces excessive smoothing.

- **Explicit Boundaries:**  
  - discriminator is never needed at runtime;
  - adversarial loss is not required for project success;
  - gradient surgery is not a default dependency.

---

### Training Curriculum Controller

- **Ownership:**  
  Controls staged training rather than enabling all losses and mechanisms at step one.

- **Stages:**
  1. structure;
  2. geometry;
  3. temporal;
  4. entity awareness;
  5. EPAM;
  6. appearance refinement;
  7. optional adversarial sharpening.

- **Explicit Boundaries:**  
  - stages reuse checkpoints;
  - no full restart unless an architecture-breaking change occurs;
  - no expensive optional stage may consume compute reserved for EPAM.

---

### ONNX Export Module

- **Ownership:**  
  Converts the chosen PyTorch inference graph into a portable deployment graph.

- **Explicit Boundaries:**  
  - training-only modules are excluded;
  - teacher model is excluded;
  - discriminator is excluded;
  - export must pass numerical parity validation.

---

### TensorRT Builder

- **Ownership:**  
  Builds the optimized target-GPU engine from ONNX.

- **Target:**
  - batch size: 1;
  - primary precision: FP16;
  - fixed or tightly bounded dynamic shapes;
  - target development resolution initially 384×216 or 512×288;
  - standalone model latency goal: under approximately 20 ms before game-loop integration.

- **Explicit Boundaries:**  
  - final engine should be built on the target RTX 5060 laptop;
  - cloud-generated TensorRT engines are not assumed portable to the target hardware.

---

### C++ Runtime

- **Ownership:**  
  Owns low-latency inference orchestration.

- **Responsibilities:**
  - TensorRT execution;
  - CUDA buffers;
  - temporal warp integration where placed outside the network;
  - entity-token packing;
  - EPAM lifecycle;
  - style-token cache;
  - performance counters;
  - fallback behavior.

- **Conceptual Loop:**
  ```cpp
  while (running) {
      engine_tick();
      capture_structure();
      build_entity_masks();
      warp_previous_features();
      prepare_epam_tokens();
      structformer_infer();
      update_epam();
      present_generated_frame();
  }
  ```

- **Explicit Boundaries:**  
  - no optimizer;
  - no training;
  - no teacher;
  - no Python requirement in the final runtime;
  - no gameplay-state mutation from neural output.

---

### Godot GDExtension Integration

- **Ownership:**  
  Bridges Godot rendering/game state and the C++ inference runtime.

- **Explicit Boundaries:**  
  - keeps neural runtime isolated from core gameplay logic;
  - does not block deterministic simulation indefinitely on renderer failure.

---

### Evaluation Harness

- **Ownership:**  
  Executes reproducible model comparisons and creates metrics, plots, and qualitative artifacts.

- **Explicit Boundaries:**  
  - no cherry-picking;
  - no modification of trained checkpoints during evaluation;
  - no model-specific scenario changes that invalidate comparison.

---

## 5. Data, Memory & State Management

### Persistence Layer

Use simple versioned filesystem artifacts rather than a database.

#### Training Data

- PNG/EXR/NPY/PT or equivalent tensor/image files;
- JSONL or Parquet manifest;
- immutable validated dataset directories.

#### Model Artifacts

- `.safetensors` or `.pt` checkpoints;
- `.onnx` deployment model;
- TensorRT serialized engine on target system.

#### Experiment Metadata

- YAML/TOML configuration;
- JSON/CSV/Parquet metrics;
- generated plots and videos.

#### Why No Database

Current dataset scale and workflow do not justify:

- operational database maintenance;
- network service dependency;
- schema migration infrastructure.

A file-based dataset is easier to reproduce, archive, inspect, and transfer across local/Kaggle/Lightning/Modal environments.

### Data Schemas

#### Training Frame

```python
TrainingFrame = {
    "schema_version": "1.0",
    "session_id": str,
    "frame_id": int,
    "timestamp_s": float,

    "structure": {
        "depth_path": str,
        "normal_path": str,
        "albedo_or_material_path": str,
        "motion_path": str,
        "entity_id_map_path": str
    },

    "camera": {
        "view_matrix": list[float],
        "projection_matrix": list[float]
    },

    "entities": [
        {
            "entity_key": str,
            "engine_class": str,
            "visible": bool,
            "spawned": bool,
            "destroyed": bool
        }
    ],

    "style": {
        "style_id": str,
        "prompt": str
    },

    "native_rgb_path": str,
    "target_rgb_path": str
}
```

#### EPAM Entry

```python
MemoryEntry = {
    "embedding": "float[D_memory]",
    "first_seen_frame": int,
    "last_seen_frame": int,
    "last_update_frame": int,
    "confidence": float,
    "style_id": str
}
```

#### Runtime Tensor Contract

```python
RuntimeInputs = {
    "structure": "float16[B,C,H,W]",
    "temporal_features": "float16[B,Ct,Ht,Wt]",
    "entity_tokens": "float16[B,K,D]",
    "entity_attention_mask": "bool/int[B,K,...]",
    "style_token": "float16[B,Ds]"
}
```

Where:

- `B = 1` at runtime;
- `K` is capped, initially recommended at 32 and later benchmarked at 64;
- padding fills unused entity-token slots.

### State Lifecycle

#### Game State

- created by Godot;
- updated only by Godot;
- destroyed on session end;
- never reconstructed from neural output.

#### Dataset

- collected;
- validated;
- versioned;
- frozen;
- consumed read-only by training.

#### Checkpoint

Naming:

```text
<component>_<stage>_step<N>.safetensors
```

Example:

```text
structformer_epam_step45000.safetensors
```

#### EPAM

```text
session starts
    ↓
empty EPAM
    ↓
entity first seen
    ↓
create memory
    ↓
high-confidence observation
    ↓
update memory
    ↓
out of view
    ↓
retain frozen memory
    ↓
reappears
    ↓
lookup memory
    ↓
destroy event
    ↓
delete entity memory

style change
    ↓
clear EPAM

session end
    ↓
clear EPAM
```

---

## 6. External Tooling & Integrations

### Third-Party APIs & Services

#### Coding Agent

Use a frontier coding agent with repository, shell, test, and file-editing capabilities.

Required workflow:

```text
specification
→ plan
→ implementation
→ compile/run
→ tests
→ inspect logs
→ repair
→ rerun
```

No architecture depends on a specific vendor.

#### Godot

Primary deterministic game engine.

#### PyTorch

Training framework for:

- StructFormer;
- losses;
- AMP/BF16/FP16;
- checkpointing;
- experiment code.

#### Frozen Geometry Feature Extractor

A pretrained 3D-aware visual feature model may provide `L_geo`.

It is:

- frozen;
- offline/training-only;
- excluded from runtime.

#### Offline Diffusion Teacher

A high-quality image-generation/stylization model may generate target RGB.

It is:

- offline;
- replaceable;
- not the final renderer;
- not needed for TensorRT deployment.

#### ONNX

Deployment interchange format.

#### TensorRT

Target NVIDIA inference runtime.

#### CUDA

Used for:

- efficient buffer management;
- optional explicit feature warping;
- asynchronous inference.

#### Compute Providers

Preferred hierarchy:

1. **Local RTX 5060 Laptop GPU**  
   Coding, smoke tests, overfit tests, ONNX/TensorRT, final inference.

2. **Kaggle or similar free GPU**  
   Baseline training, debugging, cheap ablations when quota permits.

3. **Lightning AI or similar free-credit GPU**  
   StructFormer prototypes and EPAM experiments when suitable hardware is available.

4. **Modal L40S**  
   Reserved for final/high-value training runs and critical experiments.

The project must remain portable across providers through one repository and config-driven training commands.

### Extensibility

#### Add a New Visual Style

1. define `style_id`;
2. collect/reuse structural sequences;
3. generate aligned teacher targets;
4. update style-conditioning dataset;
5. train/fine-tune;
6. evaluate geometry/temporal/entity metrics;
7. verify EPAM clears on style transition in v1.

#### Add a New Game

1. create game spec;
2. coding agent builds Godot project;
3. entity lifecycle tests run;
4. structural export tests run;
5. record sessions;
6. evaluate existing StructFormer before retraining;
7. fine-tune only if domain shift requires it.

#### Add a New Structural Signal

1. classify signal as continuous, categorical, key/address, or temporal;
2. define intended semantics;
3. add exporter;
4. add validator;
5. update schema version;
6. add ablation;
7. never append categorical identity values as continuous channels without explicit encoding rationale.

#### Replace the Offline Teacher

1. implement teacher adapter;
2. preserve target alignment;
3. version teacher metadata;
4. regenerate only required targets;
5. compare target quality before replacing existing dataset.

---

## 7. Tech Stack & Infrastructure Rationale

| Layer | Technology | Rationale & Why Chosen | Alternative Rejected |
|---|---|---|---|
| Game engine | Godot | Deterministic game state, physics, collision, open tooling | Custom engine: unnecessary risk; Unity/Unreal: more integration overhead |
| Engine native extension | C++ GDExtension | Low-overhead integration and runtime control | Python inside runtime: latency/deployment complexity |
| Coding workflow | Frontier coding agent | Multi-file build/test/repair without training a custom agent | Custom GameCoder model: expensive and unrelated to research contribution |
| Training language | Python | Best research ecosystem and PyTorch support | C++ training stack: slower iteration |
| Main renderer | StructFormer | Single-pass, transformer-friendly EPAM, real-time-oriented | Pure U-Net final model: weaker entity-token mechanism |
| Input stem | CNN | Efficient local edge/geometry processing | Tokenizing raw high-res buffers directly: inefficient |
| Spatial backbone | Window transformer | Wider context with bounded attention cost | Full global ViT: expensive |
| Entity-memory fusion | Masked cross-attention | Direct, interpretable connection from engine mask to EPAM token | Raw object-ID channel: semantically wrong |
| Temporal alignment | Engine-native motion reprojection | Uses authoritative motion, cheaper and more interpretable | RAFT production dependency: redundant estimated motion |
| Persistent identity | EPAM | Main novelty; long-horizon reappearance persistence | Short-window recurrence only: loses identity after long absence |
| Style conditioning | Cached style token | No heavy text encoder per frame | Re-encode prompt every frame: wasted latency |
| Training target | Offline diffusion teacher | High-quality supervision without runtime cost | Runtime diffusion: latency risk |
| Core training | Supervised + geometry + temporal + entity losses | More stable than adversarial-only training | GAN-only training: instability |
| Optional sharpening | PatchGAN-style adversarial loss | Can restore high-frequency detail if needed | Mandatory discriminator: unnecessary complexity |
| Baseline renderer | CNN/U-Net | Simple, fast, strong control baseline | None — baseline intentionally simple |
| One-step DiT | Future/stretch only | Potential quality ceiling | Required DiT: too risky for 2.5-month/8 GB runtime target |
| Optimizer | AdamW/Adam | Stable, standard, reproducible | Custom optimizer as default: unnecessary risk |
| Optional optimization | Structure-priority gradient balancing | Interesting ablation if time permits | Required gradient surgery: schedule risk |
| Dataset format | Files + JSONL/Parquet | Portable and reproducible | Database: unnecessary |
| Checkpoints | safetensors/PT | Standard ecosystem | Custom format: no benefit |
| Training hardware | Free GPUs + Modal L40S | Controls cost while reserving fast hardware for final runs | H100-only plan: expensive |
| Local inference GPU | RTX 5060 Laptop, 8 GB | Available target hardware | Cloud inference: defeats local real-time demo goal |
| Export | ONNX | Standard PyTorch → deployment boundary | PyTorch runtime embedding: heavier |
| Inference | TensorRT FP16 | Optimized NVIDIA C++ runtime | Pure Python inference: weak real-time integration |
| Runtime precision | FP16 initially | Strong compatibility/speed/quality balance | FP8 first: added compatibility/debug risk |
| Runtime batch | 1 | Interactive rendering | Batched inference: unnecessary latency |
| Initial resolution | 384×216 or 512×288 | Feasible performance target | 1080p from day one: unrealistic |
| Performance target | ~30 FPS end-to-end | Demonstrates interactive viability | Max-quality offline render: different project |

---

## 8. Coding Conventions & Project Structure

### Directory Layout

```text
structrender/
├── README.md
├── StructRender_KB.md
├── pyproject.toml
├── CMakeLists.txt
│
├── configs/
│   ├── data/
│   ├── teacher/
│   ├── structformer/
│   ├── training/
│   ├── eval/
│   └── deployment/
│
├── agent/
│   ├── prompts/
│   ├── specifications/
│   ├── workflows/
│   └── validators/
│
├── game_project/
│   ├── project.godot
│   ├── scenes/
│   ├── scripts/
│   ├── assets/
│   ├── instrumentation/
│   └── tests/
│
├── data_pipeline/
│   ├── recorder/
│   ├── exporters/
│   │   ├── depth/
│   │   ├── normals/
│   │   ├── material/
│   │   ├── motion/
│   │   └── entity_id/
│   ├── entity_lifecycle/
│   ├── validation/
│   ├── schemas/
│   └── teacher_targets/
│
├── renderer/
│   ├── structformer/
│   │   ├── cnn_stem.py
│   │   ├── window_attention.py
│   │   ├── encoder.py
│   │   ├── masked_entity_attention.py
│   │   ├── decoder.py
│   │   ├── rgb_head.py
│   │   └── model.py
│   │
│   ├── baselines/
│   │   ├── unet.py
│   │   └── temporal_unet.py
│   │
│   ├── temporal/
│   │   ├── warp.py
│   │   ├── confidence.py
│   │   └── fusion.py
│   │
│   ├── epam/
│   │   ├── entity_key.py
│   │   ├── memory_table.py
│   │   ├── lifecycle.py
│   │   ├── feature_pool.py
│   │   ├── token_projector.py
│   │   └── conditioning.py
│   │
│   ├── style/
│   │   ├── encoder.py
│   │   └── cache.py
│   │
│   ├── losses/
│   │   ├── reconstruction.py
│   │   ├── perceptual.py
│   │   ├── geometry.py
│   │   ├── temporal.py
│   │   ├── entity_consistency.py
│   │   └── optional_adversarial.py
│   │
│   ├── training/
│   │   ├── train_structure.py
│   │   ├── train_geometry.py
│   │   ├── train_temporal.py
│   │   ├── train_entity.py
│   │   ├── train_epam.py
│   │   └── curriculum.py
│   │
│   └── optional/
│       ├── gradient_balancing/
│       └── one_step_dit/
│
├── inference/
│   ├── export/
│   │   ├── export_onnx.py
│   │   └── validate_onnx.py
│   ├── tensorrt/
│   │   ├── builder/
│   │   └── benchmark/
│   ├── cuda/
│   │   └── temporal_warp/
│   └── cpp_runtime/
│       ├── include/
│       └── src/
│
├── eval/
│   ├── protocols/
│   ├── baselines/
│   ├── metrics/
│   │   ├── image_quality.py
│   │   ├── geometry.py
│   │   ├── temporal.py
│   │   ├── reappearance.py
│   │   └── runtime.py
│   ├── plots/
│   └── reports/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
├── scripts/
│   ├── collect_data.py
│   ├── validate_dataset.py
│   ├── generate_teacher_targets.py
│   ├── train.py
│   ├── run_eval.py
│   └── benchmark_runtime.py
│
└── checkpoints/
    └── .gitkeep
```

### Naming Conventions

- Python files: `snake_case.py`
- Python variables/functions: `snake_case`
- Python/C++ classes: `PascalCase`
- Configuration keys: `snake_case`
- No unexplained abbreviations in public module names.
- Entity key:
  ```text
  <session_id>:<instance_id>:<generation>
  ```
- Checkpoint:
  ```text
  structformer_<stage>_step<N>.safetensors
  ```
- Evaluation run:
  ```text
  <model>__<game>__<style>__<seed>__<timestamp>
  ```

### Error Handling Strategy

#### Offline

Abort or quarantine on:

- missing structural channel;
- shape mismatch;
- invalid motion;
- NaN/Inf;
- duplicated frame ID;
- invalid entity lifecycle;
- missing target;
- failed schema validation.

Never silently substitute invalid values.

#### Training

- validate config before GPU allocation;
- run a local smoke test before cloud launch;
- abort on non-finite loss after writing diagnostics;
- checkpoint frequently;
- log failing batch IDs;
- resume rather than restart whenever architecture is unchanged.

#### Runtime

Priority:

1. deterministic gameplay continues;
2. display remains usable;
3. generative quality is preserved when possible.

Fallback sequence:

```text
StructFormer frame valid
    ↓
display generated frame

single inference failure
    ↓
hold previous valid generated frame

repeated inference failure
    ↓
display native Godot RGB
```

### Logging & Observability

#### Training Log

Record:

- total loss;
- each loss component;
- learning rate;
- GPU VRAM;
- batch/step time;
- images/sec;
- checkpoint;
- validation samples;
- geometry metric;
- temporal error;
- RCS;
- EPAM create/update/freeze/delete counts;
- rejected low-confidence updates.

#### Runtime Log

Example:

```json
{
  "frame_id": 1842,
  "engine_ms": 4.3,
  "capture_ms": 2.1,
  "warp_ms": 0.7,
  "epam_ms": 0.5,
  "tensorrt_ms": 17.8,
  "display_ms": 2.0,
  "total_ms": 27.4,
  "visible_entities": 11,
  "epam_entries": 35,
  "fallback": false
}
```

---

## 9. Decision Log (ADRs - Architecture Decision Records)

### Decision 1: Deterministic Engine Over Pure Generative World Simulation

- **Context:**  
  Generated video alone cannot guarantee exact persistent game state.

- **Chosen Path:**  
  Godot owns all simulation and truth.

- **Rejected Alternatives:**  
  Pure video/world-model gameplay because it conflicts with structural correctness.

---

### Decision 2: Godot Is the Source of Truth

- **Context:**  
  A complete gameplay engine is needed without reimplementing physics/collision.

- **Chosen Path:**  
  Godot + GDExtension.

- **Rejected Alternatives:**  
  Custom engine, Unity, and Unreal for scope/integration reasons.

---

### Decision 3: StructFormer Replaces GAN-First as the Final Renderer

- **Context:**  
  EPAM maps naturally to transformer entity tokens and cross-attention. A pure CNN/GAN is simpler but less expressive for structured entity-memory conditioning.

- **Chosen Path:**  
  Lightweight hybrid CNN + window-transformer StructFormer.

- **Rejected Alternatives:**  
  Pure GAN/U-Net as final model. U-Net remains a baseline.

---

### Decision 4: GAN Becomes Optional Training Supervision

- **Context:**  
  Adversarial training may sharpen output but adds instability.

- **Chosen Path:**  
  Reconstruction/perceptual/geometry/temporal/entity losses are core. PatchGAN-style adversarial loss is optional.

- **Rejected Alternatives:**  
  Making generator/discriminator equilibrium a required success condition.

---

### Decision 5: One-Step DiT Is Not a Required Deliverable

- **Context:**  
  A one-step DiT could improve generative richness but increases training/distillation, TensorRT, VRAM, and latency risk.

- **Chosen Path:**  
  Keep DiT as future/stretch work only.

- **Rejected Alternatives:**  
  Making a custom one-step DiT the primary renderer under the 2.5-month schedule.

---

### Decision 6: Heavy Diffusion Is an Offline Teacher

- **Context:**  
  High-quality diffusion outputs are useful, but local real-time diffusion inference is unnecessary.

- **Chosen Path:**  
  Use diffusion only to generate/correct target styled frames offline.

- **Rejected Alternatives:**  
  Runtime diffusion in the core architecture.

---

### Decision 7: Engine Motion Replaces RAFT in Production

- **Context:**  
  Godot already knows deterministic movement.

- **Chosen Path:**  
  Use engine-native motion vectors/camera transforms for explicit feature warping.

- **Rejected Alternatives:**  
  RAFT as a production dependency.

---

### Decision 8: Motion Has an Explicit Operation, Not a Raw Channel Role

- **Context:**  
  Blindly concatenating motion vectors asks the network to rediscover a known geometric operation.

- **Chosen Path:**  
  Use motion to warp/reproject previous features.

- **Rejected Alternatives:**  
  Motion as an unexplained extra renderer channel.

---

### Decision 9: Entity IDs Are Addresses, Not Continuous Features

- **Context:**  
  Numeric identity values have no meaningful ordering.

- **Chosen Path:**  
  IDs create masks, lifecycle keys, and EPAM addresses.

- **Rejected Alternatives:**  
  Raw numeric object-ID channel in the production model.

- **Allowed Control:**  
  A naive raw-ID ablation may exist only as a negative/control experiment.

---

### Decision 10: EPAM Is the Headline Contribution

- **Context:**  
  Engine-grounded rendering alone is not a sufficiently sharp research claim.

- **Chosen Path:**  
  Engine-keyed persistent appearance memory.

- **Rejected Alternatives:**  
  Large temporal Transformer or multiple competing flagship novelties.

---

### Decision 11: EPAM Is Session-Local

- **Context:**  
  Cross-session memory introduces a separate persistence and identity problem.

- **Chosen Path:**  
  Clear EPAM at session end.

- **Rejected Alternatives:**  
  Cross-session persistent identity in v1.

---

### Decision 12: Style Change Clears EPAM in v1

- **Context:**  
  Appearance memory from one style can contaminate another.

- **Chosen Path:**  
  Clear memory on style change.

- **Rejected Alternatives:**  
  Style-independent identity memory before the core experiment is validated.

---

### Decision 13: No Arbitrary EPAM Eviction in the Core Benchmark

- **Context:**  
  LRU eviction would weaken the persistence claim.

- **Chosen Path:**  
  Retain all previously seen live entities for the session.

- **Rejected Alternatives:**  
  Generic cache eviction in the primary experiment.

---

### Decision 14: Window Attention Over Global Full-Resolution Attention

- **Context:**  
  Interactive rendering needs bounded attention cost.

- **Chosen Path:**  
  Hierarchical local/window attention.

- **Rejected Alternatives:**  
  Large global ViT at full spatial resolution.

---

### Decision 15: Runtime Style Encoding Is Cached

- **Context:**  
  Style prompt normally changes rarely relative to frame rate.

- **Chosen Path:**  
  Encode once and cache.

- **Rejected Alternatives:**  
  Running a text encoder every frame.

---

### Decision 16: TensorRT Engine Is Built on the Target Laptop

- **Context:**  
  Serialized TensorRT engines can depend on target hardware/runtime details.

- **Chosen Path:**  
  Export ONNX in training environment, build engine locally.

- **Rejected Alternatives:**  
  Assuming a cloud-built TensorRT engine is portable to the RTX 5060.

---

### Decision 17: C++ Owns the Final Runtime Orchestration

- **Context:**  
  Real-time game integration requires predictable latency and low overhead.

- **Chosen Path:**  
  C++/GDExtension + TensorRT + CUDA.

- **Rejected Alternatives:**  
  Python in the final frame loop.

---

### Decision 18: FP16 Before FP8

- **Context:**  
  FP16 is safer and easier to validate.

- **Chosen Path:**  
  FP16 first, optional lower precision after correctness.

- **Rejected Alternatives:**  
  FP8 as the initial deployment path.

---

### Decision 19: Train in Stages and Resume Checkpoints

- **Context:**  
  Restarting every ablation wastes compute.

- **Chosen Path:**  
  Curriculum training with checkpoint continuation.

- **Rejected Alternatives:**  
  Training every stage from scratch.

---

### Decision 20: Paid GPU Budget Is Reserved for High-Value Runs

- **Context:**  
  Cloud compute budget is limited.

- **Chosen Path:**  
  Local/free GPUs for debugging and baselines; L40S for final StructFormer/EPAM runs.

- **Rejected Alternatives:**  
  H100-first or cloud-only development.

---

### Decision 21: Entity Token Overflow Is Resolved by Visibility Ranking, Not Arbitrary Order

- **Context:**  
  `K_max` bounds entity token slots for fixed-shape TensorRT inference, but no rule previously defined which entities get dropped when more than `K_max` are visible at once — an arbitrary drop order risked silently corrupting the reappearance benchmark by dropping the entity under test.

- **Chosen Path:**  
  Rank visible entities by `pixel_count` each frame and keep the top `K_max`; dropped entities render without EPAM conditioning that frame but are not deleted from memory or treated as having left view. Core benchmark scenes are kept at or below `K_max` entities; overflow behavior is reported as a separate secondary robustness result.

- **Rejected Alternatives:**  
  Spawn-order or engine-iteration-order dropping; silently variable `K_max`; treating an overflow-dropped entity as destroyed or exited.

---

## 10. Roadmap & Milestones

### Current Phase

The architecture is considered locked. Implementation should begin.

### Milestone 0 — Repository + Minimal Godot Game

Deliver:

- repository skeleton;
- coding-agent workflow;
- minimal 3D game;
- automated launch/test;
- logging.

Exit criteria:

- project runs;
- tests execute with one command;
- agent can repair a deliberately broken scene/script.

---

### Milestone 1 — Structural Export

Deliver:

- depth;
- normals;
- material/albedo;
- motion;
- entity-ID map;
- camera matrices;
- entity events.

Exit criteria:

- synthetic scenes verify exported values;
- entity identity remains stable across leave/re-enter;
- no silent channel failures.

---

### Milestone 2 — Dataset Recorder + Validator

Deliver:

- manifest;
- frame storage;
- temporal sequences;
- reappearance sequences;
- schema versioning;
- validator.

Exit criteria:

- corrupted sessions are rejected;
- valid sessions replay correctly.

---

### Milestone 3 — Offline Teacher Targets

Deliver:

- teacher adapter;
- selected target style(s);
- cached styled targets;
- alignment validation.

Exit criteria:

- target frames preserve intended scene layout sufficiently for supervised training.

---

### Milestone 4 — CNN/U-Net Baseline

Deliver:

- frame-independent baseline;
- baseline metrics;
- ONNX export smoke test.

Exit criteria:

- model trains;
- model overfits a tiny set;
- model produces recognizable styled output;
- export works.

---

### Milestone 5 — StructFormer Baseline

Deliver:

- CNN stem;
- window transformer encoder;
- decoder;
- RGB head;
- style conditioning.

Exit criteria:

- stable training;
- better or competitive output against CNN baseline;
- ONNX export succeeds;
- **TensorRT benchmark happens here, not at the end**.

---

### Milestone 6 — Temporal Reprojection

Deliver:

- engine-guided warp;
- confidence/disocclusion mask;
- temporal loss.

Exit criteria:

- lower temporal error than frame-independent StructFormer.

---

### Milestone 7 — Structured Entity Awareness

Deliver:

- EntityKey service;
- masks;
- entity feature pooling;
- entity token projection;
- entity consistency loss;
- entity token overflow policy (visibility-ranked top-`K_max` selection).

Exit criteria:

- correct entity mask/token association;
- no raw-ID production input;
- overflow scenes (visible entities > `K_max`) drop deterministically by the defined ranking rule, never by arbitrary order.

---

### Milestone 8 — EPAM — Protected Main Novelty

Deliver:

- memory table;
- lifecycle;
- confidence-gated update;
- entity-masked cross-attention;
- style reset;
- destruction handling;
- reappearance retrieval.

Exit criteria:

- same entity retrieves same memory after leaving view;
- destroyed entity cannot leak memory to a later object;
- style change clears memory;
- EPAM improves RCS over non-memory model.

---

### Milestone 9 — Full Evaluation

Required model ladder:

| ID | Model | Backbone | Temporal | Structured Entity Use | EPAM |
|---|---|---|---:|---:|---:|
| A | CNN baseline | U-Net | No | No | No |
| B | CNN temporal | U-Net | Yes | No | No |
| C | StructFormer baseline | StructFormer | No/limited | No | No |
| D | StructFormer temporal | StructFormer | Yes | No | No |
| E | StructFormer entity-aware | StructFormer | Yes | Yes | No |
| F | **Full StructRender** | StructFormer | Yes | Yes | **Yes** |

Optional control:

| ID | Model | Purpose |
|---|---|---|
| N | Naive raw-ID input | Demonstrate that numeric ID exposure is not equivalent to structured entity use |

---

### Milestone 10 — ONNX → TensorRT → C++ → Godot

Deliver:

- ONNX parity test;
- TensorRT FP16 engine;
- C++ runtime;
- GDExtension integration;
- runtime logs;
- fallback.

Exit criteria:

- stable live loop;
- target approximately 30 FPS at the selected development resolution when hardware permits;
- TensorRT standalone latency approximately ≤20 ms preferred before integration;
- renderer failure does not break gameplay.

---

### Suggested 10-Week Sequence

| Week | Primary Goal |
|---:|---|
| 1 | Repository, Godot test game, agent build-test-repair |
| 2 | Structural buffers + entity lifecycle |
| 3 | Dataset recorder/validator + teacher targets |
| 4 | CNN/U-Net baseline |
| 5 | StructFormer + early ONNX/TensorRT benchmark |
| 6 | Temporal reprojection |
| 7 | Entity masks/tokens + EPAM core |
| 8 | EPAM training + RCS experiments |
| 9 | Ablations, optimization, final training |
| 10 | C++/TensorRT/Godot integration + report/demo |
| Remaining days | Buffer for bugs, paper, figures, optional experiments |

### Upcoming / Stretch Phase

Only after the core system and RCS result exist:

1. stronger learned occlusion confidence;
2. deeper feature-level temporal fusion;
3. structure-priority gradient balancing;
4. optional PatchGAN adversarial sharpening;
5. higher resolution;
6. one-step DiT research branch.

### Explicit Out-of-Scope

Current project will not require:

- foundation-scale world-model training;
- custom-trained coding LLM;
- renderer-controlled physics;
- generated RGB as gameplay truth;
- cross-session EPAM;
- zero-shot identity reconstruction for never-seen entities;
- visual re-identification replacing engine identity;
- mobile deployment;
- 1080p real-time as a first milestone;
- one-step DiT as a required deliverable;
- runtime diffusion teacher;
- mandatory adversarial training;
- massive global ViT;
- multi-GPU distributed training as a requirement.

---

## 11. Evaluation, Testing & Verification

### Testing Strategy

#### Unit Tests

Required:

- tensor shape validation;
- depth normalization;
- normal normalization;
- entity-ID map integer validation;
- EntityKey uniqueness;
- lifecycle generation increment;
- mask creation;
- motion warp direction;
- camera motion convention;
- confidence mask;
- EPAM create;
- EPAM update;
- EPAM freeze;
- EPAM destroy;
- EPAM style reset;
- token packing/padding;
- masked cross-attention correctness;
- finite loss;
- gradient flow;
- ONNX parity.

#### Local Pre-Cloud Training Gate

Before every expensive cloud run:

1. run 100–500 iterations locally;
2. verify loss decreases;
3. verify no NaN/Inf;
4. overfit 32–128 frames;
5. visually inspect outputs;
6. only then launch cloud training.

This gate is mandatory to protect the compute budget.

#### Integration Tests

1. Godot → structural exporter.
2. exporter → recorder.
3. recorder → dataset loader.
4. entity events → EntityKey → EPAM.
5. motion → warp.
6. StructFormer → ONNX.
7. ONNX → TensorRT.
8. TensorRT → C++.
9. C++ → Godot display.

#### End-to-End Test

```text
spawn NPC
→ observe NPC
→ initialize EPAM
→ turn camera away
→ continue simulation
→ wait
→ return camera
→ verify same EntityKey
→ retrieve EPAM
→ render
→ calculate RCS
```

### Verification Scenarios

#### Scenario 1: Static Structural Faithfulness

- fixed camera;
- deterministic scene;
- compare output against target and geometry features.

Expected:
- structure remains aligned.

#### Scenario 2: Camera Motion

- smooth camera pan;
- object remains visible.

Expected:
- temporal model flickers less than frame-independent baseline.

#### Scenario 3: Short Reappearance

- entity out of view for 1 second.

Expected:
- full model ≥ non-memory baselines.

#### Scenario 4: Long Reappearance

Test:

```text
1 s
5 s
10 s
30 s
60 s
```

This is the headline experiment.

#### Scenario 5: Destroy + ID Reuse

- view entity;
- destroy entity;
- ensure memory deletion;
- create new entity with potentially reused low-level index;
- verify new generation key prevents stale memory reuse.

#### Scenario 6: Style Change

- populate memory under style A;
- switch to style B;
- EPAM clears;
- visible entities initialize fresh style-B memories.

#### Scenario 7: Low-Confidence Observation

- heavy occlusion / very small visible area;
- confidence below threshold.

Expected:
- EPAM does not update.

#### Scenario 8: Renderer Failure

- simulate failed TensorRT call.

Expected:
- gameplay continues;
- previous/native frame fallback activates;
- error logged.

### Core Metrics

#### Image Quality

- LPIPS;
- SSIM;
- optional PSNR;
- FID only if sample count makes it meaningful.

#### Geometry

- frozen 3D-feature distance;
- edge/depth alignment where useful.

#### Temporal

- motion-warp error;
- temporal LPIPS/flicker.

#### Reappearance Consistency Score — RCS

For entity \(i\):

\[
RCS_i(\Delta t)=\cos(E(I_i^t),E(I_i^{t+\Delta t}))
\]

Report:

- mean;
- standard deviation;
- per entity type;
- per style;
- per unseen duration.

#### Runtime

- engine ms;
- capture/export ms;
- warp ms;
- EPAM ms;
- TensorRT ms;
- display ms;
- total frame ms;
- FPS;
- VRAM;
- fallback rate.

### Headline Figure

The highest-priority result is:

```text
Reappearance Consistency Score
            versus
Time Outside View
```

Comparing:

- CNN baseline;
- temporal baseline;
- StructFormer without EPAM;
- StructFormer + EPAM.

A strong EPAM result should degrade materially slower as time unseen increases.

---

## 12. Known Limitations & Open Risks

### Fragile Areas

#### Godot Structural Export

Risk:
- version/rendering-path differences;
- GPU readback/synchronization cost;
- incorrect buffer interpretation.

Mitigation:
- synthetic scenes;
- pinned engine version;
- visualization utilities;
- early instrumentation prototype.

#### Motion Coordinate Convention

Risk:
- previous/current frame direction;
- NDC vs UV;
- Y-axis inversion;
- camera-only vs object motion.

Mitigation:
- known translation tests;
- flow visualization;
- checkerboard warp test.

#### EPAM Memory Drift

Risk:
- repeated imperfect updates slowly corrupt identity memory.

Mitigation:
- confidence gating;
- freeze when uncertain;
- report update counts;
- optionally evaluate anchor-style update policies later.

#### Entity ID Reuse

Risk:
- new entity accidentally receives destroyed entity's memory.

Mitigation:
- `session_id + instance_id + generation`.

#### Style Contamination

Risk:
- old appearance token biases new style.

Mitigation:
- clear EPAM on style change in v1.

#### Teacher Structural Distortion

Risk:
- offline diffusion teacher may alter geometry and produce inconsistent targets.

Mitigation:
- geometry-preserving target generation;
- reject targets with unacceptable structure;
- use deterministic structural losses during student training.

#### ONNX/TensorRT Operator Compatibility

Risk:
- custom attention/masking operation exports poorly.

Mitigation:
- use ONNX-safe primitives;
- perform TensorRT export in Week 5;
- prefer fixed/bounded shapes;
- keep optional layers outside the critical inference graph if needed.

### Performance Bottlenecks

Likely:

1. Godot structural-buffer extraction;
2. GPU synchronization;
3. feature warp;
4. transformer attention;
5. entity-token packing;
6. TensorRT execution;
7. display copy.

Target budget for ~30 FPS:

\[
33.3\text{ ms/frame}
\]

Preferred approximate allocation:

| Component | Target |
|---|---:|
| Engine + structural export | 4–7 ms |
| Temporal warp | <1 ms |
| EPAM/token preparation | <1 ms |
| StructFormer TensorRT | 15–20 ms |
| Display/transfer | 2–4 ms |
| Safety margin | remaining budget |

These are engineering targets, not guaranteed measured results.

### VRAM Constraint

Target laptop:

- RTX 5060 Laptop GPU;
- 8 GB VRAM;
- Godot and TensorRT share GPU resources.

Therefore the inference model should preferably remain comfortably below total available VRAM and leave space for:

- Godot;
- structural buffers;
- TensorRT workspace;
- Windows/graphics overhead.

A practical goal is to design the neural runtime to use only a few GB rather than consuming almost all 8 GB.

### Compute Budget Risk

Cloud budget is limited.

Mitigation:

- local smoke tests;
- free GPU providers for low-value experiments;
- checkpoint continuation;
- reserve Modal L40S for final/high-value runs;
- avoid H100 by default;
- do not train every stage from scratch;
- do not make one-step DiT mandatory.

### Technical Debt Accepted Intentionally

- initial low resolution;
- session-local EPAM;
- style-reset memory;
- simple confidence mask before learned confidence;
- fixed/annealed loss weights;
- no dynamic unlimited entity-token count;
- fixed `K_max` entity slots (overflow resolved by visibility ranking, see Decision 21 — the debt accepted is the fixed cap itself, not an undefined drop rule);
- no one-step DiT in the main branch;
- no cross-game canonical identity.

These are deliberate scope controls.

---

## 13. Glossary

- **StructRender:** The full engine-grounded neural rendering system.
- **StructFormer:** The primary lightweight hybrid CNN + window-transformer renderer.
- **Godot source of truth:** Principle that all authoritative gameplay state comes from Godot.
- **Structural buffer:** Deterministic per-pixel scene information such as depth, normals, and material/albedo.
- **G-buffer:** General term for structural render buffers produced by the engine.
- **Motion vector:** Per-pixel or deterministically derived displacement between frames.
- **Temporal reprojection:** Warping previous features into current-frame coordinates using engine motion.
- **Disocclusion:** Newly visible area with no valid previous-frame correspondence.
- **Temporal confidence mask:** Mask controlling where warped history may safely be reused.
- **Entity ID map:** Per-pixel categorical map indicating which engine entity owns a pixel.
- **EntityKey:** Session-unique stable identifier, normally `<session_id>:<instance_id>:<generation>`.
- **Entity mask:** Binary spatial mask for one deterministic engine entity.
- **EPAM:** **Engine-Keyed Persistent Appearance Memory**, the main novelty.
- **Appearance memory:** Compact generated feature representation associated with an EntityKey.
- **Appearance embedding:** Vector stored by EPAM.
- **Entity token:** Transformer-compatible projected representation of an EPAM memory entry.
- **Entity-masked cross-attention:** Attention in which engine masks control which spatial tokens can access which entity memories.
- **Entity feature pooling:** Aggregating features inside an entity mask to produce a compact entity representation.
- **Confidence-gated memory update:** EPAM update allowed only when the current observation is reliable.
- **Within-session persistence:** Scope of EPAM; memory does not survive the gameplay session.
- **Style reset:** Clearing EPAM when the active style changes in v1.
- **Offline teacher:** Heavy image-generation/stylization model used to create training targets only.
- **Diffusion teacher:** Offline diffusion model acting as the teacher target generator.
- **CNN/U-Net baseline:** Simple frame-independent reference renderer.
- **Temporal baseline:** Model that uses deterministic motion consistency but no EPAM.
- **Naive raw-ID control:** Deliberately weak control that treats numeric IDs directly, used only to prove that structured ID use matters.
- **Window attention:** Attention performed inside bounded spatial windows to reduce computational cost.
- **StructFormer encoder:** Hierarchical transformer feature encoder.
- **StructFormer decoder:** Transformer reconstruction path producing output features.
- **Style token:** Cached representation of the active visual style.
- **RCS:** **Reappearance Consistency Score**, measuring how similar the same entity looks before and after time outside view.
- **Geometry loss:** Frozen-feature loss penalizing structural distortion.
- **Temporal loss:** Motion-warp consistency objective.
- **Entity consistency loss:** Loss encouraging the same engine entity to retain similar appearance features across observations.
- **Optional adversarial loss:** PatchGAN-style training loss that may sharpen output; not part of runtime.
- **One-step DiT:** One-step Diffusion Transformer; future/stretch renderer, not required.
- **Curriculum training:** Sequential training from structure → geometry → temporal → entity → EPAM → appearance refinement.
- **Build-test-repair loop:** Coding-agent workflow requiring actual execution and repair before acceptance.
- **ONNX:** Intermediate exported inference representation.
- **TensorRT:** NVIDIA optimized inference runtime used by the final C++ application.
- **FP16:** Half-precision floating-point mode used as the first deployment target.
- **GDExtension:** Godot's native-extension interface used to integrate the C++ runtime.
- **Runtime fallback:** Holding the last valid generated frame or using native Godot RGB if inference fails.
- **Structural drift:** Generated scene geometry changing inconsistently.
- **Appearance drift:** Same entity being rendered with inconsistent identity across observations.
- **Teacher target:** Offline styled ground-truth frame used to supervise StructFormer.
- **K_max:** Fixed maximum number of visible entity tokens packed for TensorRT inference. Overflow (more visible entities than `K_max`) is resolved deterministically by keeping the top `K_max` entities ranked by `pixel_count`; see Entity Token Packing & Overflow Policy and Decision 21.
- **Core path:** Features required for project success.
- **Stretch goal:** Feature that may be removed without invalidating the main research contribution.
