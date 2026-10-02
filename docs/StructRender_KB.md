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
│   │   ├── dataset.py            # M4 (Decision 35)
│   │   ├── train_baseline.py     # M4 (Decision 35)
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

### Decision 22: M0 Logging Autoload Is Named `GameLog`, Not `Logger` (2026-09-03)

- **Context:**  
  `agent/specifications/m0_minimal_game.md` names the logging autoload `Logger`. Godot `4.7.2-stable` ships a built-in native `Logger` class (`ClassDB.class_exists("Logger") == true`). In GDScript the native class name shadows the autoload's global identifier, so any script referencing `Logger.log_event(...)` fails at parse time with `Static function "log_event()" not found in base "GDScriptNativeClass"`, and the autoload itself fails to instantiate.

- **Chosen Path:**  
  Register the M0 logging autoload as `GameLog` (script still `game_project/scripts/logger.gd`). All engine-side references use `GameLog`. The M0 spec's event schema, file layout (`user://logs/session_<unix_ts>.jsonl`), and NDJSON format are unchanged.

- **Rejected Alternatives:**  
  Keeping the name `Logger` (blocked by the engine); using `class_name` gymnastics or `Engine.get_singleton` lookups everywhere (adds noise for no benefit); downgrading Godot (engine version is pinned for structural-export reasons).

- **Follow-up:**  
  `agent/specifications/m0_minimal_game.md` should be updated to say `GameLog`; M1+ instrumentation must not reintroduce a `Logger` autoload.

---

### Decision 23: Structural-Export Render Path Is Pinned to Forward+ (2026-09-03)

- **Context:**  
  §12 "Fragile Areas → Godot Structural Export" names "version/rendering-path differences" a fragile area, but the KB never fixed a render path. Milestone 0's `game_project/project.godot` had to pick one; it sets `rendering/renderer/rendering_method = "forward_plus"` and also overrides the mobile method to `forward_plus`. Milestone 1's depth/normal/albedo/material/motion exporters and Milestone 10's runtime capture will be written against whatever path M0 established, so leaving it as an undocumented in-file choice risks a silent mismatch later.

- **Chosen Path:**  
  Pin the structural-export render path to **Forward+** for the current project horizon. Godot's Forward+ renderer exposes the G-buffer-style data (depth prepass, normal-roughness, per-pixel motion vectors) the structural exporters need; the Mobile and Compatibility paths do not expose the same buffers consistently. The `.mobile` override to `forward_plus` exists so a device/feature-tag fallback cannot silently switch the project onto an export-incompatible path.

- **Rejected Alternatives:**  
  Mobile / Compatibility render paths (missing or inconsistent structural buffers); leaving the render path unpinned (defeats the §12 "pinned engine version" style of mitigation and invites M1 exporter drift).

- **Scope:**  
  Changing the render path after this point is an architecture-breaking change for the data pipeline — it requires a new dated entry here and re-validation of every structural exporter, not a quiet `project.godot` edit.

---

### Decision 24: Test / Instrumentation Autoloads May Ship in `game_project`, Guarded (2026-09-03)

- **Context:**  
  Milestone 0 ships `SimDriver` — a headless input-driving, auto-quitting controller — as a project autoload in `game_project`, gated by a `structrender_sim` feature tag / headless check and a per-frame `Engine`-meta opt-out (`structrender_no_sim`). Milestones 1–2 will add more instrumentation autoloads (structural capture, entity-lifecycle recording) into the same shipped project. The KB had no convention for this.

- **Chosen Path:**  
  Instrumentation and test-support autoloads may live in `game_project`, but must be **inert by default in a normal windowed play session** — activated only by an explicit feature tag, headless detection, or an engine-meta / CLI flag — and must never write to `GameState` or otherwise become a source of gameplay truth (§4 Game Runtime boundary). Each such autoload documents its activation condition in its own file header.

- **Rejected Alternatives:**  
  A separate instrumented copy of the game project (duplication, drift between play and capture builds); unconditional instrumentation autoloads (risk of altering a normal play session, and of an input driver running during real play).

- **Note:**  
  `.claude/agent-memory/godot-engine/godot47_headless_testing.md` records that `set_physics_process(false)` is not reliably sticky for a node that defines `_physics_process`; instrumentation that needs a hard off-switch should use an explicit per-frame guard, not rely on disabling processing.

---

### Decision 25: Motion Vector Convention Pinned (2026-09-03)

- **Context:**  
  §12 flags "Motion Coordinate Convention" as the top fragile area (previous/current direction, NDC vs UV, Y-axis inversion, camera-only vs object motion). Milestone 1 computes motion vectors; leaving the convention implicit invites warp-direction bugs in Milestone 6 (temporal reprojection), where the sign of the field silently determines whether history is pulled from the right place.

- **Chosen Path:**  
  Motion is **backward** (frame `t` → frame `t-1`), **screen-UV normalized `[0,1]`**, **Y-down** (same axis as all structural buffers), includes **camera + rigid-object** motion, and is computed **analytically from engine transforms** (unproject current depth → apply `M_{t-1} M_t^{-1}` → reproject with the previous-frame camera), never estimated from pixels (consistent with Decision 7/8). Full definition in `data_pipeline/schemas/frame_structure.md` §4.

- **The three "no valid previous correspondence" cases all resolve to `motion = (0,0)`, never a guess:**  
  1. **Frame 0** (no `t-1` at all) — whole-frame zero buffer, plus `motion_valid = false` on the frame.  
  2. **Background / sky** (`depth == +inf`) — per-pixel zero. Not reprojected at the far plane for camera-only motion: zero-history background is discarded by the M6 disocclusion confidence mask regardless, so far-plane camera motion buys nothing and adds a parallax failure mode.  
  3. **Entity with no `t-1` transform** (spawned or first registered this frame) — per-pixel zero for that entity's pixels. It recovers correct analytic motion on the next frame once it has a previous transform. `motion_valid` stays a whole-frame flag; these per-pixel zeros are not separately flagged in the buffer.

- **Rejected Alternatives:**  
  Forward motion (t → t+1, needs future state); NDC-space or pixel-space units (resolution-coupled, and NDC Y is up while the buffers are Y-down — a guaranteed sign bug); camera-only motion (breaks on moving entities); pixel-estimated flow (redundant, non-deterministic, violates Decision 7); reprojecting background at the far plane (adds a failure mode for zero benefit — see above); falling back to identity for a new entity's `M_{t-1}` (produces a geometrically meaningless non-zero vector — the original M1 bug this clause fixes).

- **Verification:**  
  Synthetic scenes S2 (known camera translation, sign + direction), S3 (known object translation, moving vs static pixels), S4 (checkerboard backward-warp photometric alignment), S7 (entity spawned after frame 0 → zero motion that frame, correct analytic motion the next frame).

---

### Decision 26: M1 Structural Export Is Override-Shader Re-Render, Not Native G-Buffer Readback (2026-09-03)

- **Context:**  
  Decision 23's rationale for pinning Forward+ mentions it "exposes the G-buffer-style data … the structural exporters need". The Milestone 1 exporter does **not** read Godot's internal Forward+ buffers. It renders the scene into a `SubViewport` several times with every mesh's `material_override` swapped for an unshaded pass shader (depth = eye-space Z, world normal, flat linear albedo, packed entity-id), reads each back with `get_image()`, and computes motion analytically on the CPU from stored per-entity + per-camera transforms. This was chosen because GDScript has no clean, version-stable access to the Forward+ G-buffer in Godot `4.7.2`, and the override-shader path is directly verifiable (S1–S7).

- **Chosen Path:**  
  Keep the override-shader multi-pass approach for the **offline data pipeline** (M1 export, M2 recorder). It is correct and testable; readback cost and the ~8 SceneTree frames advanced per `export_frame` call are acceptable offline (KB §12 already lists readback cost as a known risk).

- **Forward constraints this creates:**  
  - **M2 recorder** must drive the exporter as an isolated capture mode — never inside a live play loop — because each captured frame advances the SceneTree ~8× and briefly swaps every material. Determinism is preserved only if the sim is stepped under the recorder's control (consistent with Decision 24).  
  - **M10 runtime capture** (live inference loop, ~30 FPS budget) **cannot** use an 8-pass re-render per displayed frame. It needs a different mechanism — a `CompositorEffect` / `RenderingDevice` tap of the native Forward+ buffers, or a single MRT pass. This is explicitly deferred to M10 and is expected to require its own validation against the M1 `frame_structure.md` contract (the contract is the invariant; the production mechanism is not).

- **Rejected Alternatives:**  
  Blocking M1 on a `RenderingDevice`/compositor G-buffer implementation (fragile, version-coupled, hard to verify — wrong risk to take for the offline pipeline); assuming the M1 mechanism carries through to the runtime (it does not — flagged here so M10 budgets for the rework).

---

### Decision 27: M2 Dataset Storage Format and Schema Extensions (2026-09-03)

- **Context:**  
  §5 *Training Frame* defines the manifest record but not the on-disk buffer format, and the M1 exporter's own `.npy` debug dumps are uncompressed full-`float32`. A recorded dataset of thousands of 512×288 frames × 5 channels needs a compact, reproducible-by-content format, and the validator needs a couple of signals §5 does not carry.

- **Chosen Path:**  
  - **Buffer storage:** one compressed `.npz` per frame (`depth`, `normal`, `albedo`, `motion` as `float16`; `entity_id` as `int32`), native RGB as an 8-bit sRGB `.png`. For `depth`/`normal`/`albedo` this is lossless: the M1 exporter renders those three into a half-float GPU target (Decision 26), so `float16` in the dataset is exactly what the exporter ever had; `entity_id` stays exact. **`motion` is the exception** — it is computed analytically on the CPU as `float32` (Decision 8, `frame_structure.md` §4), so storing it as `float16` is a deliberate precision reduction. It is bounded and harmless for the training use: for typical screen-UV displacements (≪ 1) the half-float ULP is sub-pixel; the only pixels where it degrades are the large-magnitude disocclusion pixels at reappearance boundaries (Decision 29 cites ~64 UV), and those are exactly the pixels M6's confidence/disocclusion mask discards. If a future stage needs full-precision motion everywhere, bump the dtype (see §7 / bump policy). This is the **dataset** format — the M1 `.npy` debug dumps are unchanged, and `data_pipeline/schemas/frame_structure.md` still governs every channel's value semantics, ranges, and conventions.
  - **Reproducibility:** buffer *content* is bit-exact across re-recordings of the same scenario; the `.npz` ZIP *container* is not byte-identical (`ZIPPacker` stamps per-member mtime). Determinism tests compare decoded arrays, not container bytes.
  - **Schema extensions beyond §5** (both in `dataset_manifest.md`): `tick` (true sim tick per frame — so a reduced recording rate is explicit in the data, not a silent `frame_id` gap) and `motion_valid` (the M1 whole-frame flag from `frame_structure.md` §4 — without it the validator cannot tell a frame-0 "no previous frame" zero-motion buffer from a real one). The `{session_id, instance_id, generation}` triple is stored alongside §5's `entity_key` string so consumers never parse the string.
  - **Namespacing:** dataset sessions are `ds_<unix_ts>_<8 hex>` under dataset root `data_pipeline/recorder/` (env `STRUCTRENDER_DATASET_ROOT` overrides). Deliberately disjoint from the `GameLog` debug log's `session_<unix_ts>` under `user://logs/` (Decision 22) — the two artifacts share no files and no id namespace, per the M2 carry-over note.

- **Rejected Alternatives:**  
  Raw `.npy` per channel (≈4× the size, no gain); `float32` storage for all channels (the exporter never had that precision for depth/normal/albedo; for `motion` the extra precision only matters on disocclusion pixels that M6 masks out — not worth ~2× on that channel); a single monolithic archive per session (defeats streaming / partial reads); reusing the debug-log `session_` namespace (conflates the gameplay log with the dataset — explicitly forbidden by the M1 review).

- **Bump policy:**  
  `schema_version` is `m<major>.<minor>`, at session and frame level. Minor = backward-compatible addition; major = dtype/shape/path/convention change or the arrival of real targets (`has_targets: true`). A validated dataset is immutable (§5) — a major bump is a new dataset, not an in-place migration. Each bump gets a dated Decision Log entry.

---

### Decision 28: M2 Seed Reappearance Ladder Is a Training Fixture, Distinct from the Eval Headline Ladder (2026-09-03)

- **Context:**  
  §11 Scenario 4 fixes the headline eval ladder at unseen durations {1, 5, 10, 30, 60} s. Recording that at full rate is ~tens of GB for a single 60 s rung. Milestone 2's job is a working recorder + validator, not the eval dataset.

- **Chosen Path:**  
  The M2 seed dataset uses a compressed proof-of-coverage ladder **{0.5, 1, 2, 4, 8} s**, both `camera_away` and `entity_away` mechanisms, 1–2 target entities, single `native` style — on the order of ~100 MB (990 frames, deflate-compressed; measured 99 MB), regenerable, gitignored, exercising every recorder and validator code path. Gaps ≤ 2 s are recorded at full rate; the 4 s / 8 s gaps may be recorded at a reduced rate **during the out-of-view span only** (boundary frames always exact, every frame carries true `tick`/`timestamp_s`, so `unseen_seconds` stays exact) — a **seed-fixture-only expedient**.  
  The full §11 {1, 5, 10, 30, 60} s ladder **and** the eval-dataset during-gap recording policy are deferred to the **eval-harness subagent**, to be settled before the scaled collection run (post-M3).

- **Rejected Alternatives:**  
  Recording the full {1,5,10,30,60} s ladder at M2 (tens of GB, and it prematurely locks the during-gap policy that belongs with the eval harness); leaving reappearance coverage to incidental play (the headline RCS-vs-time-unseen figure needs the durations covered by construction).

---

### Decision 29: M2 Dataset Validator Acceptance Rules (2026-09-03)

- **Context:**  
  `dataset_manifest.md` is the recorder-side contract; several of its "the validator checks X" lines and the KB M2 "Notes carried from M1" leave the *method* and *thresholds* to the validator. Milestone 2 exit criteria (① corrupted sessions rejected, ② valid sessions replay) need those pinned so a pass/fail verdict is reproducible. The validator lives at `data_pipeline/validation/` (module + `python -m data_pipeline.validation` CLI); it only reports and writes the `VALIDATED` marker — it never repairs or mutates a session.

- **Chosen Path:**  
  - **Partial channel dropout** (`dataset_manifest.md` §6.1) is detected by strict per-pixel channel cross-agreement, tolerance **0** by default (`--dropout-tolerance` overrides). The four signals, all of which are exactly 0 on every frame of the seed session: `entity_id != 0 ∧ depth == +inf`; `entity_id != 0 ∧ normal == (0,0,0)`; `depth finite ∧ normal == (0,0,0)`; `depth == +inf ∧ normal != (0,0,0)`. Rationale: the M1 override-shader re-render (Decision 26) produces all five channels from one geometry pass, so a non-background `entity_id` implies rendered geometry implies finite depth, and `normal` is exactly `(0,0,0)` iff `depth` is `+inf` (`frame_structure.md` §2.2). `albedo` is deliberately **not** a dropout signal — a legitimately black material is indistinguishable from a dropped albedo pixel.  
  - **Motion magnitude is unbounded.** The validator checks `motion` is finite but imposes no magnitude ceiling: analytic backward reprojection of a surface that was far off-screen during a fast camera swing legitimately produces screen-UV displacements ≫ 1 (observed up to ~64 at reappearance boundaries in the seed session). `frame_structure.md` §4 never bounded it; consumers must not clamp it.  
  - **`motion_valid` on a temporal window's first frame must be `false`.** A temporal window is a dense standalone clip (`dataset_manifest.md` §4/§5) with no in-window history, so its first frame is treated like frame 0. `motion_valid == false` additionally requires an all-zero motion buffer; `motion_valid == true` on `frame_id 0` is a fault.  
  - **Derived reappearance fields are recomputed, not trusted.** `last_seen_frame` / `first_seen_again_frame` must equal the actual last/first visible frames bracketing the (single, contiguous) unseen gap; `unseen_frames` and `unseen_seconds` must equal `tick` / `timestamp_s` differences of those two boundary frames exactly; `generation_before/after` must equal the target entity's `generation` at those frames and match `identity_stable`. `gap_ticks_configured` mismatch is a **warning**, not a failure (`dataset_manifest.md` §4 marks it "cross-check only").  
  - **Debug-log separation** (`dataset_manifest.md` §6.2) is enforced structurally: the session dir may contain only `manifest.json`, `manifest.jsonl`, `events.jsonl`, `frames/`, and (validator-written) `VALIDATED`; `frames/` may contain only `frame_NNNNN{.npz,_rgb.png,_camera.json}`; any `session_*` name or `logs/` artifact is rejected as `GameLog` namespace leakage.  
  - **`VALIDATED` marker** is JSON (validator version, UTC timestamp, git commit, session id, schema version, frame count, checks run, dropout tolerance, warnings). Written only on a full clean run (never for a `--limit` partial run); re-validation overwrites it idempotently.  
  - **Test fixture policy:** corrupted-session tests are built programmatically by copying the gitignored seed session (a prefix slice ending on a sequence boundary for speed, plus one full-session replay test) and mutating one thing. The seed session dir name carries a per-recording timestamp (`ds_<unix>_<hex>`, Decision 27), so the tests **discover** it — `$STRUCTRENDER_SEED_SESSION`, else the newest `ds_*` dir under `$STRUCTRENDER_DATASET_ROOT` (default `data_pipeline/recorder/`) — and read every asserted frame count / session id back from its own `manifest.json`; nothing is hardcoded to one recording. When no seed session is present, `scripts/test_validator.sh` **exits 2 = SKIPPED** (distinct from PASS) and `scripts/test.sh` reports it as SKIP, never as a pass — an all-skipped pytest run must not masquerade as verification of exit criterion ①. `scripts/test_validator.sh` is wired into `scripts/test.sh` after the recorder step.

- **Rejected Alternatives:**  
  A fractional dropout threshold (would mask small dropped patches, which are exactly the failure mode the per-frame exporter guard can't see); using `albedo` as a dropout signal (false-positives on black materials); a motion magnitude ceiling (rejects valid fast-camera reprojection); trusting the manifest's `unseen_*` / boundary fields instead of recomputing (defeats the point — they are derived, per §4/§5); an LRU/eviction or auto-repair path in the validator (out of scope and forbidden — validator reports only).

---

### Decision 30: M2 Loose Ends — Validator Tolerances, Native-RGB Capture Settings, `--limit` Mode (2026-09-03)

Recorded from the M2 spec-guardian review; none of these change a locked decision, they pin choices the earlier M2 entries left implicit.

- **Validator numeric tolerances** (`data_pipeline/validation/validator.py`), alongside the dropout tolerance of Decision 29: unit-normal acceptance band `0.9–1.1` on finite-depth pixels (consistent with the exporter's `NORMAL_LEN_TOL = 0.1`); per-component `|normal| ≤ 1.02`; `albedo ≥ -0.01`; inline-vs-sidecar camera-matrix agreement `1e-4`; `timestamp_s == tick / tick_rate` within `1e-6`; `unseen_seconds` derived-equality within `1e-6`. These are half-float / float32 round-trip slop, not semantic thresholds; a channel value outside them is a real fault. If a future channel dtype bump (§7) changes the round-trip error, revisit these with it.
- **Native-RGB capture settings** are a recorder-side choice with M3 alignment implications (§3C, §4 Offline Teacher): `dataset_recorder.gd` renders the native (unstyled) frame through its **own** `SubViewport` + `Camera3D` cloned from the source camera, with MSAA off, TAA off, screen-space-AA off, positional shadow atlas 0, `use_hdr_2d = false` → RGBA8 → 8-bit sRGB PNG. It is captured **before** the exporter swaps `material_override` on every mesh, so it is frame-aligned with that frame's structural buffers by construction. M3's teacher adapter must consume this exact image and must not assume different capture settings; a change here is a schema-affecting change (dated entry + bump).
- **`--limit N` validator mode** (`python -m data_pipeline.validation … --limit N`) inspects only the first N frames' buffers. It is a debugging shortcut, never a replay: `report.replayable` is false and the `VALIDATED` marker is never written (already implied by Decision 29's "never for a `--limit` partial run").
- **Provenance decision list:** the recorder's `provenance.kb_decisions_in_force` now also lists Decision 9 (entity IDs are addresses) and Decision 29 (validator acceptance rules), which its entity-id handling and the validator contract materially depend on.

---

### Decision 31: M3 Teacher Method, Style Scope, and Sidecar Target Storage (2026-09-05)

- **Context:**  
  §10 Milestone 3 names four deliverables (teacher adapter, selected target style(s), cached styled targets, alignment validation) and one qualitative exit criterion — "target frames preserve intended scene layout sufficiently for supervised training". Decision 6 fixes the teacher as *offline diffusion* but names no model; §12 flags *Teacher Structural Distortion* as a live risk without a rejection rule; and Decision 27's bump policy says the arrival of real targets (`has_targets: true`) is a **major** bump / "a new dataset", which collides with the M2 structural session being **immutable and VALIDATED**. All three gaps are settled here.

- **Chosen Path:**  
  - **Teacher = Stable Diffusion 1.5 + ControlNet-depth img2img**, driven by the dataset's own exported depth buffer rather than an estimated depth map — structural preservation is a *conditioning input*, not a hope. Fits the local 8 GB card (the teacher is offline and is **not** bound by the runtime 8 GB budget per §4, but fitting locally avoids a cloud dependency at M3). Licences: SD 1.5 CreativeML OpenRAIL-M, ControlNet Apache-2.0 — both acceptable for research use and to be attributed in the repo. Generation is seed-pinned and deterministic on a fixed machine but **not bit-exact across GPU/driver**, which is precisely why targets are cached and version-keyed rather than regenerated on demand (§3C.5).
  - **A second, dependency-free `deterministic` backend** (classical edge-preserving / posterising filters over the native RGB) ships behind the same adapter interface as the CI and test teacher. It is perfectly aligned and bit-exact, so the adapter contract, cache, and alignment-validation code paths are all testable on a clean checkout with no model download. It is **not** a research-grade teacher and must never be the source of targets used to report results.
  - **Style scope: one real style at M3, but every path is style-keyed from day one** — directory layout, target manifest `style_ids`, adapter signature, cache key. Unlike the M2 reappearance ladder (Decision 28), which needed re-recording and was therefore genuinely cheaper to settle early, targets are a **derived sidecar over shared structural data**: adding style B later costs one generation run and *zero* code change. EPAM's style-reset behaviour (Decision 12, §11 Scenario 6) is not exercised until M8, so nothing before then requires the second style. A single-style dataset must not be allowed to bake in single-style assumptions — the "add a second style by config alone" path is a **tested** deliverable at M3, not an aspiration.
  - **Targets live in a sidecar dataset**, `data_pipeline/teacher_targets/<style_id>/<session_id>/`, with its own manifest referencing the immutable m2.0 structural session by `session_id` + `frame_id`. This **interprets, and does not override, Decision 27's bump policy**: a validated structural session is never mutated, never re-recorded, and never duplicated per style. `has_targets` / `target_rgb_path` in the *structural* manifest stay as recorded at M2; a consumer joins structure to targets through the sidecar manifest. The schema-major-bump rule of Decision 27 continues to apply to the **structural** dataset only.

- **Rejected Alternatives:**  
  An in-engine second-style render pass (perfectly aligned and free, but a low quality ceiling and not the "high-quality offline teacher" Decision 6 means — it would weaken the supervision M4/M5 learn from); a Blender path-traced re-render (highest quality and geometrically exact, but needs a Godot→Blender scene export plus per-frame transform replay — a multi-week pipeline, disproportionate to a milestone whose job is to unblock M4); classical stylisation as the *primary* teacher (kept, but only as the test/CI backend); two styles at M3 (no cheaper now than later given the sidecar design, and 2× generation cost for capability nothing needs until M8); a new `m3.0` dataset with targets inlined (the literal reading of Decision 27, but it duplicates ~99 MB of structural data per style and would require re-recording or copying the one artifact whose bit-exact frame alignment makes the targets valid at all).

- **Open at time of writing:**  
  The alignment-validation pass/fail thresholds are **not** locked by this entry. Starting values are boundary F-measure ≥ 0.60 @ 2 px against the engine structural edge map (depth discontinuity ∪ `entity_id` boundary), spurious-edge ratio ≤ 0.35, and per-entity silhouette support ≥ 50 % for entities ≥ 200 visible px. These are to be **calibrated against the observed metric distribution on real generated targets** and then pinned in their own dated entry — guessing thresholds before seeing the distribution produces a gate that passes everything or nothing.

---

### Decision 32: M3 Alignment Gates — Four, Not Three; Calibrated Thresholds (2026-09-05)

- **Context:**  
  Decision 31 left the alignment thresholds explicitly unlocked pending calibration against real generated targets. Calibration produced two findings that changed the gate *set*, not just its numbers.

- **Finding 1 — three gates were not enough.**  
  Gates 1–3 all ask "did the structure survive". The perfect way to satisfy every one of them is to change nothing at all: measured on the seed session, the **untouched native frame** scores `boundary_f = 0.976`, `spurious = 0.000` — better than any real target. So "lower the denoise strength until it passes" converges on a teacher that does nothing, and the M3 exit criterion would be satisfied by a passthrough. **Gate 4 (`align.not_stylised`)** puts a floor on how far the target moved from the native frame, measured as colour-histogram L1 distance (the histogram term carries the gate because the scene is dark: a real restyle shifts the palette far while moving mean |Δpixel| only ~2 levels of 255). `test_passthrough_native_frame_is_rejected` locks this in and asserts it is *that* gate doing the work, not a structural one.

- **Finding 2 — the first teacher config failed on motion.**  
  At `strength 0.65 / controlnet 1.0`, 39 of 990 frames failed the hallucination gate — **every one inside the moving-camera window**, none in the ladder. Stronger conditioning fixes them decisively (frame 61: spurious `0.545 → 0.000`), and on genuinely varied content it costs almost no stylisation. Pinned at **`strength 0.45 / controlnet_conditioning_scale 1.3`**. Going further (`0.35 / 1.5`) also passes but trades away restyling for alignment already in hand.

- **Chosen Path — gates, and the evidence they were calibrated against.**  
  Measured over the full 990-frame validated set (`ds_1788624547_cf0ceeda`, `gouache`, real teacher), **0 frames failing**:

  | gate | threshold | observed min / p05 / median / max |
  |---|---|---|
  | 1 `boundary_f` | ≥ 0.60 | 0.709 / 0.878 / 0.980 / 1.000 |
  | 2 `spurious_edge_ratio` | ≤ 0.35 | 0.000 / 0.000 / 0.000 / **0.337** |
  | 3 `min_silhouette_support` | ≥ 0.60 (entities ≥ 200 px) | 0.661 / 0.818 / 0.993 / 1.000 |
  | 4 `stylisation_hist` | ≥ 0.15 | 0.232 / 0.310 / 0.509 / 0.955 |

  Supporting parameters: `tolerance_px 2`, `edge_threshold 0.008`, `strong_edge_threshold 0.04`, `depth_rel_jump 0.05`, `normal_angle_deg 30`, `albedo_jump 0.05`. `max_fail_frac` stays **0.0 (strict)** — one failing frame rejects the set; the knob exists for a documented exception, not as a default cushion.

  **Every threshold was fixed before this data was seen** (they are Decision 31's starting values, except gate 3 which moved 0.50 → 0.60 during calibration and gate 4 which is new). This is a genuine pass, not a threshold fitted to the result.

- **Known limitation — gate 2's margin is thin.**  
  The worst frame scores 0.337 against a 0.35 limit: **0.013 of headroom**. A busier scene, a more painterly style, or a longer recording could cross it legitimately. When that happens the correct response is to re-examine whether the edges are genuinely unsupported (as was done here — the `reference` vs `support` map split exists precisely because scoring painterly strokes along texture boundaries as hallucinated geometry produced false positives), **not** to raise the number. Any change to gate 2 needs its own dated entry with the evidence.

- **Rejected Alternatives:**  
  Relaxing gate 2 to ~0.55 so the original 39 failures passed (fits the threshold to the data and stops the gate catching real hallucination); raising `max_fail_frac` to ~0.05 to absorb them (would have hidden a 24 %-of-distinct-content failure rate behind duplicate frames — see Decision 33); pixel-domain SSIM/LPIPS against the native frame (punishes the stylisation we asked for); measuring gate 1's precision on the permissive edge set (rejects the real teacher while accepting a 3 px misaligned one).

---

### Decision 33: Seed Sessions Must Carry Visual Diversity; Target Cache Is Content-Addressed (2026-09-05)

- **Context:**  
  Found while validating the first full M3 target set. The M2 seed session `ds_1788430292_4c77250c` contains **163 distinct images across 990 frames** — three static frames account for **830 of them** (295× / 295× / 240×). Cause: `dataset_harness.gd::run_rung` reset the camera and target to fixed `cam_home` / `target_home` poses and then held them frozen through each rung's pre-margin, out-of-view gap, and post-margin. The reappearance *semantics* were correct throughout (Decision 29's recomputed `last_seen` / `first_seen_again` / tick deltas all validated), so M2's validator had no reason to object — it checks structural and sequence correctness, and never asked whether the frames differ from each other.

- **Why it matters:**  
  It silently invalidates downstream measurement rather than breaking anything visibly. M3's alignment failure rate reads 39/990 = 3.9 %, but every failure falls in the 160-frame moving-camera window, so the rate on *distinct* content is 39/163 ≈ 24 %. Worse, M4's exit criterion "model overfits a tiny set" would pass **vacuously** against a set that is 84 % three images, as would any claim about temporal or reappearance learning. A dataset defect that makes success criteria easier to meet is the most dangerous kind, because nothing fails.

- **Chosen Path:**  
  - **Seed recording keeps the scene moving.** `record_still` / `run_rung` apply a small, bounded, tick-driven drift (`_apply_drift`) to the camera and entities through *every* phase, gap included. Amplitudes sit far inside the frustum margin so the target's visibility state is never altered — the property all reappearance semantics rest on — and per-mechanism exclusivity is preserved for the pair `dataset_manifest.md` §4 actually constrains, **the target entity and the camera**, for the whole gap: under `camera_away` the target is frozen and the camera moves; under `entity_away` the camera is frozen and the target moves. **Other entities keep drifting under either mechanism** — they are not the subject of the sequence and their motion does not enter its semantics; the seed scene's second entity does exactly this. Drift is a pure function of the sim tick, so re-recording stays deterministic.
  - **A seed session is not fit for purpose on sequence correctness alone.** Any future recorder change must be checked for *frame diversity*, not just schema and sequence validity. Distinct-frame count is the cheap proxy (hash the native RGB); a session whose distinct fraction collapses is a defect even when every validator gate passes.
  - **Target cache is content-addressed** (`cache.content_key`): keyed on a hash of the teacher's actual conditioning inputs (native RGB + depth) plus `teacher_version` and `seed`, in addition to the per-frame `(session_id, frame_id, style_id, teacher_version, seed)` key. Identical inputs generate once and are reused, with `content_key` / `content_source_frame` recorded per manifest line. This is **not an approximation**: `sd15_controlnet.py` seeds one generator per *style*, not per frame, so identical inputs under the same version and seed are guaranteed to produce identical output — the reused render is exactly what regeneration would have written. **If a backend is ever changed to seed per frame, this equivalence breaks and content reuse must be disabled with it.** Not primed from the existing index under `--force`, which must regenerate rather than silently reuse what it was asked to replace.

- **Rejected Alternatives:**  
  Accepting the duplicate-heavy seed and documenting it only (leaves M4's overfit criterion meaningless and defers the debt to a worse moment); regenerating targets for the defective dataset first and fixing the recorder later (pays the expensive generation twice and measures M3 against content that cannot support the claim); regenerating only the frames that failed a gate (fastest, but produces a target set whose frames were made with different generation parameters than the manifest records — a reproducibility regression in the one artifact whose entire purpose is cached, versioned, reproducible supervision).

---

### Decision 34: M3 Review Follow-ups — Measurement Limits, Enforced Rules, Code Location (2026-09-05)

Recorded from the M3 spec-guardian review. Nothing here overrides a locked decision; each item either pins a limit that was measured but undocumented, or replaces a rule that existed only as prose with something the code enforces.

- **Alignment translation-detection floor.** The gates do not catch a small *purely horizontal* target shift on this scene, because its dominant structure (the ground/backdrop horizon) is a long horizontal line and therefore near-invariant under horizontal translation. Measured floors: **3 px diagonal, 8 px x-only** — at 5 px x-only the worst frame still scores `F = 0.717`, `spurious = 0.294`, `silhouette = 0.630`, inside every gate. **A systematic sub-5 px horizontal misalignment would pass all four gates and reach training as supervision.** Recorded in `teacher_targets.md`, `metrics.py`, and pinned by `test_purely_horizontal_shift_is_rejected`.
  > **⚠ SUPERSEDED BY DECISION 41.** These floors are scene-bound and were re-measured on the enriched scene as **2 px diagonal / 5 px x-only** — the values now in `metrics.py`, `teacher_targets.md` and `test_alignment_rejects.py`. This bullet's 3 px / 8 px describe the retired scene only.
  > This bullet also ended, as originally written, with: *"A scene with more vertical structure would **raise** the floor; do not assume this number transfers."* The second clause stands; the first had the **direction backwards**. More vertical structure **lowers** the floor — a lower floor means smaller misalignments are caught, which is the improvement Decision 38 predicted and Decision 41 measured. The original sentence is quoted here rather than deleted so the correction has something to correct.
- **Partial circularity in the alignment measurement.** The teacher is conditioned on the exported depth buffer, and gate 1's recall is measured against a reference map derived from that same buffer. The `support` map adds normal and albedo discontinuities, which are *not* conditioning inputs, so this is partly mitigated — but these metrics **cannot detect a structural distortion that ControlNet-depth is itself blind to**. Breaking the circularity would need a signal the teacher never sees; deferred, and it is a real limit on what "alignment PASS" asserts.
- **Gate 3 does not cover every frame.** `min_silhouette_support` is defined only where at least one entity has ≥ `min_entity_px` visible pixels: on the seed session that is 695 of 990 frames, so **~30 % of frames are ungated on entity silhouette**. Decision 32's table reads as 990 samples per row; this is the correction.
- **Frame diversity is now a check, not a sentence.** Decision 33 required future recorder changes to be checked for frame diversity but left that to whoever remembered. It is now `diversity.low` in the M2 validator (`_check_frame_diversity`), hashing native RGB and reporting `distinct_frames` / `distinct_frac` / `most_repeated_frame` into the `VALIDATED` marker. Emitted as a **warning**, not an error — a short or deliberately static session is legitimate and this validator's errors mean "structurally invalid" — with the threshold at `--min-distinct-frac 0.5`. Verified against both sessions: the retired seed warns at 163/990 (16.5 %, 295× repeat), the current one is clean at 990/990.
- **Content-addressed reuse is gated by a backend capability flag.** Decision 33 stated that a backend seeding per frame breaks reuse's correctness and must disable it, but nothing bound the two. `TeacherBackend.CONTENT_ADDRESSABLE` now does, **defaulting to `False`** so a new backend opts in deliberately; `generate.py` reads it off the class (no model load) and skips the reuse path entirely when it is unset. Both current backends set it: `deterministic` has no per-frame randomness, `sd15_controlnet_depth` seeds one generator per *style*.
- **An unvalidated structural session is now an error, not a warning.** `ALIGNED` asserts the targets are usable supervision; that claim is unsound over a base that never passed M2, since every alignment metric is computed from that session's own buffers. It also made the M3 validator laxer than the M2 one, which refuses to write `VALIDATED` for anything short of a clean full run.
- **Code location:** §8 lists `scripts/generate_teacher_targets.py`; the implementation is the package `data_pipeline/teacher/` driven by `python -m data_pipeline.teacher`, matching how the M2 validator is laid out (`data_pipeline/validation/`) rather than adding a loose script. `data_pipeline/teacher_targets/` remains the *artifact* root as §8 and Decision 31 specify. `check_layout.sh` enforces the actual layout.
- **Retired seed session.** `ds_1788430292_4c77250c` (the 163-distinct-frame defect) is kept on disk for comparison but is superseded by `ds_1788624547_cf0ceeda`. It is gitignored and regenerable. Fixture discovery takes the newest `ds_*`, so it is inert — but anything that pins `$STRUCTRENDER_SEED_SESSION` to it is running against a session Decision 33 retired, and it now self-identifies through the `diversity.low` warning.

---

### Decision 35: M4 Baseline — Input Channels, Sequence-Holdout Split, Pre-Fixed Overfit Thresholds (2026-09-05)

Pins the acceptance rules Milestone 4 introduces. Nothing here overrides a locked decision; each item makes an M4 exit criterion reproducible instead of a judgement call. Config: `configs/training/m4_unet_baseline.json`; tests: `renderer/training/tests/`; runner: `scripts/test_baseline.sh`.

**Revised 2026-09-06 after the M4 spec-guardian review.** Four things in the original entry were wrong or overstated rather than merely incomplete, and are corrected in place below: the overfit set was N = 8, **below §11's mandatory 32–128 range**; the quoted overfit numbers were from a run the test suite had since overwritten; "recognizable styled output" rested on a metric that rewards blur; and LPIPS was recommended as the ladder headline without its do-nothing anchor ever being measured (it is the *worst* of the three, not the best). The superseded text is not preserved — the corrections state what was wrong.

- **Input is 7 channels: depth (1) + normals (3) + albedo (3). Motion and entity IDs are out.**
  This is §4 *StructFormer CNN Stem* read literally ("normalized depth; normals; material/albedo features", and its Explicit Boundaries: "raw entity IDs are not concatenated here; motion is not treated as an unexplained appearance channel"), plus Decision 8 (motion has an explicit warp operation, arriving at M6) and Decision 9 (IDs are addresses). The ladder position is what makes this non-negotiable rather than merely tidy: §11's Model **A** is Temporal *No* / Structured Entity Use *No*, and the optional Model **N** exists *only* to show that raw-ID input is not equivalent to structured entity use. An ID channel in Model A would delete that control. Asserted by `test_entity_id_is_not_an_input_channel` (permuting `entity_id` must not change the built input) and `test_motion_is_not_an_input_channel`.

- **Depth normalization is inverse depth, `d_inv = 1 / (1 + d)`.**
  `depth` is raw linear eye-space metres with `+inf` for background (`frame_structure.md` §2.1). `1/(1+inf) == 0` exactly, so the background sentinel lands at the bottom of the range with no magic constant, no clamp and no scene-dependent far-plane normalisation; the result is bounded `[0,1]` and monotone in nearness. Rejected: min/max or near/far normalisation (couples the network to one scene's depth range and needs a separate background mask), and `1/d` (unbounded as `d -> 0`).

- **Train/val split is by contiguous sequence with a guard band — never by random frame.**
  All 990 frames come from ONE session at 60 fps with continuous drift (Decision 33 deliberately made neighbouring frames *non-identical*, not *independent*), so a random frame split puts a val frame's immediate neighbours in train and reports a leak as a result. Held out: the two 8 s reappearance rungs `reap_camera_away_8s` [492, 574] and `reap_entity_away_8s` [907, 989] — 166 frames. Additionally `guard_frames = 60` (1.0 s of drift) of train frames on each side of every held-out span are **dropped from both splits**, so train and val are non-adjacent, not merely disjoint: 644 train / 166 val / 180 dropped. `test_train_and_val_are_non_adjacent` asserts the minimum train-to-val frame distance exceeds the guard. The split is recorded in every metrics report and in the checkpoint. Measured representativeness: entities are visible in 107/166 val frames (64 %) vs 437/644 train (68 %) — the reappearance rungs do not bias the held-out set toward empty scenes.

- **The overfit exit criterion's thresholds were fixed before the first training run; N = 32 is §11's mandated floor.**
  N = **32** frames (evenly spaced across the training span, not the first 32 — consecutive frames would make the criterion nearly vacuous), max 2000 steps, batch 2, lr 2e-4 **pinned in the `overfit` block** (not inherited from `train.lr`, so retuning the training LR cannot silently move an exit criterion), seed 20260905. Pass requires **all** of: PSNR ≥ 30 dB, final L1 ≤ 0.02, final L1 at least 10× below the step-0 loss, SSIM ≥ 0.95. They live in the config and are asserted one-per-test so a failure names the criterion. This is the direct answer to Decision 33's warning that "a dataset defect that makes success criteria easier to meet is the most dangerous kind" — the same applies to a criterion chosen after seeing the result.
  **N was 8 in the first M4 pass, which violated §11's *Local Pre-Cloud Training Gate* item 4 ("overfit 32–128 frames", "mandatory to protect the compute budget"). 8 frames are strictly easier to memorise than 32, so that run was a weaker test than the gate requires.** Re-run at N = 32 against the *same* unrelaxed thresholds (2026-09-06): **39.58 dB / 0.004307 / 78.5× / 0.99294 — all four pass.** For reference the retired N = 8 run scored 42.18 dB / 0.003326 / 102.3× / 0.99418.
  > **⚠ RETIRED-SCENE FIGURES.** Every number in this bullet was measured on `ds_1788624547_cf0ceeda` with the retired `gouache` style and an L1-only objective. They are historical (Decision 38: M4 is *superseded, not invalidated*). **`eval/reports/m4_overfit.json` no longer contains them** — it holds the current N = 32 run on `ds_1788687350_337657ea`, recorded in **Decision 42**. Deliberately no figures here: this sentence exists to stop a reader misreading that file, and re-typing its numbers is how it goes stale. Read the artifact. Do not cite this bullet's numbers or read them against that file.
  The **loss-drop factor is a sanity check, not one of four independent gates.** The head is sigmoid-initialised (output ≈ 0.5) against dark targets, so step-0 L1 is large by construction and clearing 10× is close to vacuous — it measured 78–102×. PSNR, final L1 and SSIM carry the verdict.

- **"Recognizable styled output" is quantified, not eyeballed — and the quantification must be blur-aware.** On the selection split the prediction must (a) be closer to the teacher target than the engine's own **unstyled** native render is, (b) travel at least 0.8× the target's colour-histogram distance from the native palette — the quantity M3's gate 4 uses to reject a passthrough (Decision 32, Finding 1) — and (c) keep ≥ 0.9 of the target's high-frequency energy. Measured **on the retired scene/style, L1-only objective**: 40.10 dB vs the native frame's 35.03 dB; histogram travel 0.561 vs the target's 0.548; `sharpness_ratio` **0.9926**. ⚠ Historical — the current session/style/objective gives 33.28 vs 29.95 dB, travel 0.555 vs 0.601, `sharpness_ratio` **0.9303** (Decision 41). Native RGB is used here **only** as an evaluation reference and never as a model input.
  **(b) alone is not sufficient, and this is a correction to how M3's gate was carried into M4.** The histogram term rises under smoothing as much as under restyling — a blurred frame keeps its shifted palette — so `hist_pred 0.561 > hist_target 0.548` was equally consistent with "blurrier than the target". Gate 4 was calibrated against a *teacher* whose failure mode is doing nothing; a trained regressor's failure mode is blur, which that metric rewards. Hence (c): `sharpness_ratio = grad_energy(pred)/grad_energy(target)`, high-pass (squared Sobel) energy on luma. Unlike the overfit thresholds, 0.9 and 0.8 were set **after** the model existed; 0.9 is calibrated against the failure mode rather than the model's score — Gaussian-blurring the teacher targets gives ratios 0.977 / 0.897 / 0.761 / 0.654 at σ = 0.5 / 1.0 / 1.5 / 2.0, so 0.9 means "no softer than a ≈ 1 px blur of the target".
  **Metric finding worth carrying forward:** the *mean absolute* Sobel magnitude — the obvious first choice — is **non-monotone under blur on this content**: it rises (1.00 → 1.04 → 1.14 → 1.19 at the same σ), because the frames are mostly flat 8-bit-quantised regions where the gradient is exactly zero and blurring spreads the few real edges over more pixels. A mean-magnitude sharpness gate here would have been backwards, not merely weak. Pinned by `test_gradient_energy_falls_monotonically_under_blur`.

- **The held-out split doubles as the model-selection set — quantified, and now labelled as such.**
  `train_baseline.py` checkpoints on the lowest `val_l1` and the reported metrics are then computed with that checkpoint, so they are best-of-N-epochs on the same frames rather than a clean generalisation estimate. With 990 frames from one session there was no room to carve a third guarded block without starving training, so the effect is **measured** instead, per run, and reported inline in `m4_metrics.json`'s `selection_set_note`. `_selection_note()` derives it from the run's own history precisely because a hardcoded one published the retired scene's gain into a new session's report and understated the real bias by ~2.5× (Decision 42). It does not affect any M4 exit criterion — none of the four need an unbiased generalisation estimate — but it is not negligible either: on the current session it is **~3.2 %**, against the ~1.3 % this bullet used to assert. Those two figures are quoted to make the *size of the earlier error* legible, and are historical the moment training is re-run. An earlier version of this sentence licensed them by asserting training is "seed-deterministic and reproduces exactly" — **that is not established.** `train_baseline.py` seeds both loops but calls neither `torch.use_deterministic_algorithms(True)` nor `cudnn.deterministic`, and training additionally uses bf16 autocast, worker processes and shuffling, i.e. more sources of nondeterminism than the overfit loop that Decision 42 measures as drifting. No test pins training reproducibility and nobody has run it twice to compare. **The safeguard is the derivation, not the determinism**: `m4_metrics.json`'s `selection_set_note` is computed by `_selection_note()` from the run's own history, is authoritative, and self-updates. Read it rather than this bullet.
  The number being small does not license calling it something it is not, so the report key is **`selection_set`, never `held_out`**, and carries a `selection_set_note` stating the bias inline; `test_reported_metrics_are_labelled_as_the_selection_set` fails if `held_out` reappears. **Ladder protocol (§11):** every model on the ladder must use the *identical* selection rule (checkpoint on lowest `val_l1` over the same split at the same eval cadence), or a genuinely untouched test split must be carved out first. A headline number may not be quoted as *generalisation* under the current setup — only as "best epoch on the selection split, same rule for every model".

- **KB §11's Required "finite loss" and "gradient flow" unit tests live at M4**, since M4 is the first milestone with a loss and a backward pass: `renderer/training/tests/test_loss_and_gradients.py`, pure CPU on synthetic tensors with no session, no targets, no checkpoint and no GPU — so they run on a clean checkout, where every data-backed M4 test skips. §11 pre-cloud gate item 3 ("verify no NaN/Inf") is additionally enforced *inside* the training loop by `assert_finite_loss`, which raises `NonFiniteLossError` and aborts; it never clamps, substitutes or skips the step. Gradient flow is asserted as "every parameter has a **non-zero** finite gradient", not merely a non-`None` one — an all-zero gradient over a whole tensor is what a severed skip connection looks like and a finite-loss check cannot see it.

- **Test runs must not overwrite committed evidence.** The first M4 pass had `test_overfit.py` and `test_onnx_export.py` write straight into `eval/reports/`, so any verification run destroyed the artifact this Decision cites — which is how the KB came to quote overfit numbers (42.49 / 0.0035 / 96× / 0.9944) that no committed file reproduced. Both tests now write to a pytest tmp path by default; the committed report is refreshed only under the explicit opt-in `STRUCTRENDER_REFRESH_REPORTS=1`. **Every figure in this Decision not marked ⚠ retired-scene is either the one in the committed artifacts or is cited by threshold and artifact rather than by value** (Decision 42) — this entry describes the *retired* session `ds_1788624547_cf0ceeda`, so most of its measurements are historical by construction.

- **Qualitative strips must include moving-camera frames.** The val split is the two 8 s reappearance rungs, which hold the camera still for most of their length, so six strips drawn only from it are near-static and under-report what the model does under motion. `eval/reports/m4_qualitative/` now carries both: `frame_*` (six evenly-spaced val frames) and `trainmove_*` (the six training frames with the largest frame-to-frame view-matrix delta, ≥ 30 frames apart — deterministic, no seed). The moving-camera strips show the model smoothing out the speckled brush detail the teacher paints along the horizon, which the aggregate `sharpness_ratio` dilutes because most of the frame is flat ground and sky. ⚠ The strips in `eval/reports/m4_qualitative/` were regenerated on the current scene/style/objective, so this observation must be re-read against them; the ratio itself is in `m4_metrics.json` and is quoted, with its do-nothing anchor, in Decision 41.

- **Loss at M4 was `L_rec` (L1) alone.**
  > **⚠ SUPERSEDED BY DECISION 41.** M4's shipped objective is `1.0·L1 + 0.5·LPIPS(alex)`. This bullet is kept because its *consequence* is what forced the change, not because it still describes the milestone. Read Decision 41 for the objective actually in force; the config (`configs/training/m4_unet_baseline.json` -> `loss`) is authoritative over both.

  As originally written: no perceptual, geometry, temporal or entity term, and no adversarial loss (Decision 4). Consequence to honour later: whatever loss the ladder's later models train under, **Model A must be re-run with the same one** or the §11 comparison is confounded by the objective rather than the architecture. That consequence came due immediately — the enriched scene made the L1-only model fail the sharpness gate — and Decision 41 is the re-run it demanded.

- **ONNX export, opset 17, fixed shapes `(1,7,288,512) -> (1,3,288,512)`, parity tolerance `1e-3` max-abs.** §12's mitigation applied from M4 so the path is proven before M5 sends custom attention/masking down it. Parity is asserted against the **tolerance**, and the measured max |onnxruntime − PyTorch| lives in `eval/reports/m4_onnx_export.json` (`parity.max_abs_diff`) — it is model- and scene-dependent and was quoted here at a retired-scene value. It clears the tolerance by more than two orders of magnitude. **Operator finding:** the TorchScript exporter never emits the native `GroupNormalization` op — at opset 17, 18 and 20 alike it lowers every `nn.GroupNorm` to `Shape -> Reshape -> InstanceNormalization -> Reshape -> Mul -> Add`, putting one `Shape` node per norm layer in the graph. All TensorRT-supported, so nothing needed working *around*, but that is where to look if a shape-free graph is ever required. The export test carries an explicit operator allow-list so a new op in M5's graph shows up as a visible diff.

- **LPIPS is the reference implementation or it is absent.** `lpips` (AlexNet trunk) is an optional extra because it pulls `torchvision` and fetches AlexNet weights on first use. When it is missing the reports say `lpips_implementation: unavailable` and carry SSIM/PSNR only. There is deliberately no fallback approximation — a home-made number published under the name "LPIPS" is worse than no number.

- **Measurement caveat for the ladder — and a correction: LPIPS is *not* the fix.** The first version of this bullet said PSNR's range is too small on this scene and recommended LPIPS as the headline. That asserted LPIPS's range without measuring it. It is now measured: `style_acquisition` reports a **do-nothing anchor for every metric** — the score the *unstyled* engine frame gets against the teacher target — because that anchor is what bounds each metric's usable range.

  > **⚠ RETIRED-SCENE TABLE** (`ds_1788624547_cf0ceeda`, `gouache`, L1-only). Superseded by Decision 41's ledger on the current session; kept because the *method* — always pair a metric with its do-nothing anchor — is what carried forward.

  | metric | do-nothing (native) | M4 baseline | perfect | fraction of the do-nothing→perfect gap the model closes |
  |---|---|---|---|---|
  | PSNR | 35.03 dB | 40.10 dB | ∞ | +5.07 dB = 3.22× lower MSE |
  | SSIM | 0.98988 | 0.99304 | 1.0 | **31.2 %** |
  | LPIPS (alex) | 0.04254 | 0.04181 | 0.0 | **1.7 %** |

  **LPIPS compresses harder than PSNR or SSIM here, not less.** On this dark, low-contrast scene the `gouache` teacher target is already perceptually almost indistinguishable from the native frame (LPIPS 0.043 between them), so LPIPS separates the trained model from doing nothing by 0.0007 — noise. Making it the ladder headline would have produced a figure on which Models A through the full EPAM model are all within a rounding error of the unstyled engine. `sharpness_ratio` tells the same story: 0.993 for the model vs 1.041 for the native frame.
  **Revised recommendation:** do not elevate any single metric on this session/style — the problem is the scene, not the metric. Report all three *always paired with their do-nothing anchor* (or as the gap-fraction in the table above), which is the only form in which they are interpretable. The real fix before the §11 ladder is a headline scene/style with more stylisation contrast than `gouache` on this dark exterior; **the headline result is RCS versus time-outside-view, which does not depend on this at all** — the image-quality ladder is the secondary figure. Recorded as a limit on what M4's numbers can support, not as a task for M5.

---

### Decision 36: M3's Stylisation Gate Is More Permissive Than It Looked — Strengthen the Style Before M5 (2026-09-06)

- **Context:**  
  Surfaced by M4, not by M3's own validation. Decision 32 added gate 4 (`stylisation_hist >= 0.15`) specifically so a teacher that does nothing could not pass, and the `gouache` target set cleared it comfortably (median 0.509, min 0.232). Training a model against those targets then produced the anchors that gate could not see:

  | measured on the M4 eval split | do-nothing (native) | teacher target |
  |---|---|---|
  | LPIPS vs target | **0.0425** | 0 |
  | SSIM vs target | **0.98988** | 1.0 |
  | PSNR vs target | **35.03 dB** | ∞ |
  | gradient energy vs target | **1.041** (native is *sharper*) | 1.0 |

  The unstyled engine frame is already perceptually near-identical to its own teacher target. Visually the "gouache painted concept art" target is the same washed-out grey render with slightly different shading.

- **Why gate 4 passed anyway:**  
  A colour-histogram L1 moves a long way under a **palette shift** while the image stays perceptually almost unchanged. Gate 4 measures *displacement in colour statistics*, which is a good detector of the failure mode it was designed against (a literal passthrough, distance exactly 0) and a poor measure of *how much the image actually changed*. The gate is not wrong — it is answering a narrower question than its name suggests, and 0.509 does not mean "half-way restyled".

- **Contributing cause:**  
  Decision 32 pinned `strength 0.45` precisely to stop the hallucinated geometry that failed 39 frames at 0.65. Lower img2img strength keeps the output near its input by construction, so alignment and stylisation trade off directly against each other. The scene compounds it: two grey cubes on a grey floor under flat ambient light, with a substantial fraction of frames near-empty, gives a style little to act on.

- **Chosen Path:**  
  **Strengthen the style before M5.** §11's ladder compares models on output quality; with this little style signal that axis cannot discriminate Model A from full StructRender, and "StructFormer paints the frame in the chosen style" would be asserted rather than demonstrated. Regenerate the M3 targets with a visually distinct prompt and/or higher strength, re-run all four M3 alignment gates, and re-tune if hallucination returns. M4's implementation is style-agnostic and unaffected; its **trained checkpoint and reported metrics must be regenerated** against the new targets before they are used as the ladder's Model A.

- **Consequences to honour:**  
  - **Report image-quality metrics against their do-nothing anchor, always.** A bare "LPIPS 0.0418" is uninterpretable; the informative quantity is the fraction of the native→target gap closed. On this data that is SSIM 31.2 %, PSNR 3.2× MSE, **LPIPS 1.7 %** — so LPIPS compresses hardest here, and Decision 35's original recommendation of LPIPS as the ladder headline is **withdrawn** (see the revised bullet there).
  - Gate 4's threshold is not re-tuned by this entry. If a future style is rejected by gates 1-3 while gate 4 is comfortable, that is the alignment/stylisation trade-off of Decision 32 doing its job, not a reason to relax either.
  - The RCS headline (reappearance consistency vs time-unseen) does not depend on style strength; this affects the image-quality axis of the ladder, not the project's main claim.

- **Rejected Alternatives:**  
  Accepting the weak style and proceeding to M5 (cheapest, but leaves the ladder's quality axis near-uninformative and the styled-rendering claim unevidenced); enriching the game scene as well (addresses the root cause that there is little to style, but reopens M0/M2 and is a large detour — kept available if a stronger style alone proves insufficient); adding the deferred second style now instead (Decision 31 makes it config-only and M8 needs it eventually, but it does not fix the reference style the ladder is measured on).

---

### Decision 37: Scene Geometry, Not the Teacher Pipeline, Is What Limits Style Strength (2026-09-06)

- **Context:**  
  Decision 36 called for strengthening the reference style before M5, because the `gouache` targets were perceptually near-identical to the unstyled engine frame and left §11's ladder almost nothing to compare models on. That attempt was run to completion and **failed**. The finding is worth more than the attempt.

- **What was tried:**  
  A six-candidate prompt/parameter sweep (cel-shaded, ukiyo-e woodblock, neon, oil-impasto, each at varying `strength` / `guidance_scale` / `controlnet_conditioning_scale`), scored on style signal — `LPIPS(native, target)`, i.e. how far the target actually moves from the engine frame — *and* all four M3 alignment gates simultaneously. Two candidates were taken to full 990-frame generation and validation.

- **Result — every configuration with more style signal than `gouache` fails the gates:**

  | config | style signal (LPIPS vs native) | full-990 verdict |
  |---|---|---|
  | oil-impasto `s0.50 g9.0 c1.4` | highest of the sweep | **FAIL — 34 frames** |
  | oil-impasto `s0.50 g7.5 c1.6` ("v2") | **0.0823** (1.6× gouache) | **FAIL — 10 frames** |
  | `gouache` `s0.45 g7.0 c1.3` (incumbent) | 0.0512 | PASS, worst frame 0.337 vs a 0.35 gate |
  | oil-impasto `s0.45 g8.0 c1.5` ("v3") | **0.0459 — *below* gouache** | passes comfortably (0.101) |

  The only variant that passes with real margin does so by restyling *less* than the style it was meant to replace. Style signal and structural fidelity are in strict opposition on this scene, and `gouache` already sits at the frontier — it passes with 0.013 of headroom.

- **Root cause is the scene, not the teacher.**  
  Every failure clusters in the same place: the spans where the main target has left the frame and entity pixels fall from ~3,500 to ~1,450 (`reap_entity_away_2s` and `_8s` gaps). With little real geometry for ControlNet-depth to anchor on, any prompt pressure strong enough to restyle the scene is also strong enough to invent brushwork in the empty regions — which is exactly what §12's hallucination gate is built to reject (worst observed frame: 93 % of strong edges unsupported). Two grey cubes on a grey floor under flat ambient light, with a substantial fraction of near-empty frames, gives a style almost nothing to act on and gives the structural conditioning almost nothing to hold.

- **Chosen Path:**  
  - **The style decision is HELD, not made.** `gouache` remains the single committed style; M4's committed checkpoint and metrics (commit `85ca946`) stay valid and untouched. No `oil_impasto` config or target set is kept — a config that fails the gates must not sit in the repo waiting to be picked up.
  - **The blocker to resolve first is the test scene.** The minimal scoped change is: more varied object types and silhouettes, genuinely distinct materials/albedo, and real depth variation, so both the style *and* the structural conditioning have something to work with. This is deliberately a small, bounded enrichment of the existing scene — not a new game — and it reopens M0/M2 (re-record the seed, re-run M3) by design.
  - **Re-run the style sweep after the scene change, not before.** The sweep methodology and metrics are sound and reusable; only the content they ran against was insufficient.

- **Method note worth carrying forward:**  
  A 40-frame stride-25 probe *passed* the first failing configuration, because it sampled frames 775 and 800 — both fine — and stepped over the failing band between them. **A uniform sample cannot find a clustered failure.** Full-set validation is what caught both failures, exactly as it caught M3's original 39. Subset probes are for ranking candidates cheaply; they are never evidence that a configuration passes.

- **Rejected Alternatives:**  
  Shipping "v2" on its 14-frame result (10 frames fail at full scale — the subset trap again); relaxing gate 2 or raising `max_fail_frac` to absorb the failures (fits the gate to the result and disables the §12 guard the failures are correctly firing); accepting "v3" (passes, but restyles *less* than the incumbent, so it costs a regeneration to move backwards); proceeding to M5 on `gouache` regardless (leaves the ladder's image-quality axis near-uninformative — the problem Decision 36 opened, still unsolved and now explained).

---

### Decision 38: Structural Albedo Never Reached the Render — Scene Enrichment Scope (2026-09-06)

- **Defect found while scoping Decision 37's scene work, and it predates that work.**  
  `structural_harness.gd::add_entity` / `add_prop` set **only** the `structrender_albedo` meta. They never assign a material. The meta feeds the structural albedo *pass*; the **native RGB** — the sole image the teacher consumes, captured before any `material_override` swap (`dataset_recorder.gd`) — therefore renders every mesh in Godot's default grey. Measured on `ds_1788624547_cf0ceeda`:

  | entity | structural albedo | native RGB |
  |---|---|---|
  | 1 | (0.90, 0.65, 0.20) — orange | (0.66, 0.66, 0.66) — grey |
  | 2 | (0.25, 0.55, 0.85) — blue | (0.67, 0.67, 0.67) — grey |

  The scene has been authored with colour since M1 and has never rendered it.

- **Consequences:**  
  - **Three of M4's seven input channels are actively misleading.** The albedo channels describe colours that appear nowhere in the target, so the trained baseline must learn to ignore them; the "7-channel structural input" is effectively four useful channels. M4's committed numbers (`85ca946`) are unaffected in validity — the model simply learned around it — but the channel budget was not what it appeared.
  - **This is a substantial part of Decision 37's root cause.** "The scene gives a style almost nothing to act on" is partly because *everything renders the same grey*, not because the scene was authored without colour.
  - `frame_structure.md` §2.3 defines a meta → material → default *precedence* for the structural pass but never requires the structural albedo to **agree** with what the native render shows. They have disagreed by default for every mesh in every synthetic scene. That gap is closed here: the harness authors both consistently (`material.albedo_color` = sRGB of the linear meta colour), with meta still taking precedence for the structural buffer.

- **Chosen Path — scene enrichment scope, ordered by value:**
  0. **Real materials in the shared harness** (`structural_harness.gd`), so the `albedo` parameter means what its name says. Shared with M1's S1–S7; their structural assertions are unaffected because meta wins for the structural pass, but S1–S7 **must be re-run** to confirm.
  1. **6–8 static, non-entity props** (`entity_id == 0`) placed to add depth discontinuities and silhouettes in the frames where entities are sparse. Three placement constraints, all load-bearing:
     - they must cover **both** the home frustum **and** the 165° aimed-away camera pose, or `camera_away` gaps — half the observed failure cluster — stay as empty as they are now;
     - they must sit **outside the camera↔target sightline across the whole drift envelope**. Visibility is `pixel_count > 0`, so a prop that *fully* occludes the target for even one frame inside a margin creates a second hidden→visible transition. Partial occlusion is fine; this must be asserted, not eyeballed;
       > **⚠ Correction (Decision 44).** "the M2 validator rejects the session (Decision 29's 'single, contiguous' gap)" was wrong. `_check_sequences` locates the **first** hidden stretch and checks that one is clean (`reap.dirty_gap`); a *second* hidden run after reappearance is not an error there. The guard is now `_check_one_hidden_run` in `d1_recorder.gd`, which counts visible→hidden transitions across the rung's frame range and requires exactly one.
     - they must sit roughly **3–40 m** from the camera. M4's input is `1/(1+d)`, so a prop at 40 m maps to 0.024 against a background sentinel of exactly 0 — beyond that, "varied depth" is numerically indistinguishable from sky, worse still in the dataset's fp16 storage.
  2. **Distinct per-prop albedo** — *subsumed by change 0*. Without real materials it raises structural-albedo variance and the `support` edge map while leaving the native RGB and therefore the teacher target unchanged, i.e. more input variance with no target variance. Worse than doing nothing.
  3. **Stepped/tilted floor sections.** Highest structural payoff per line: a floor crease produces both a depth jump and a normal discontinuity, so it lands in `R` *and* `S`, and in `camera_away` gaps the floor may be the only geometry in frame.

- **Explicitly cut** (each fails the bar "does it add depth edges, silhouettes, or albedo boundaries where entities are sparse?"): textures/normal maps (inert — the flat-albedo override pass never sees them); more entities (the failure is when entities are *absent*, and it pressures M7's `K_max`); lighting/shadows (structural passes are unlit, so a shadow edge is a strong native-RGB edge with **no** counterpart in `R` or `S` — gate 2 would score it as hallucinated geometry, the exact opposite of the goal; `add_light_env` must keep shadows disabled); animated props (no structural gain, risks Decision 33's determinism).

- **Measurement discipline — the enrichment makes the gates easier, so aggregates prove nothing.**  
  Adding always-visible props inflates `n_R` with easy edges that ControlNet reproduces trivially, shrinking the entity silhouette's share of `boundary_f` and diluting a hallucination *on the entity*; a larger `S` mechanically raises supported-strong coverage. This is precisely Decision 33's "a defect that makes success criteria easier to meet". Therefore: after re-recording, compare gate distributions old-vs-new **restricted to the previously failing spans** (`reap_entity_away_2s` / `_8s` gaps), never on the aggregate. Gate 3 resists the dilution but does not cover it — Decision 34 records that ~30 % of frames carry no entity ≥ 200 px and are ungated on silhouette.

- **Correction to Decision 37's stated mechanism:** the second-order risk was described there as changing gate 2's denominator. It does not — `spurious_edge_ratio`'s denominator is the **strong target-image** edge count, not the reference-edge count. The dilution risk is on **gate 1's recall**, whose denominator is `n_R`. The conclusion (re-verify, do not assume thresholds transfer) stands; the reasoning is corrected here.

- **Probe redesign, stated before running (Decision 37's method note):** the follow-up style sweep's wide probe is stratified **by failure mode, not by frame index** — (a) a fixed regression set of every frame that failed either previous run, carried forward permanently; (b) the bottom decile by reference-edge density, sampled exhaustively; (c) contiguous blocks per sequence rather than strided points, because failures cluster; (d) uniform spread only as additional coverage. Unchanged: a subset probe ranks candidates; only full-990 validation is evidence of a pass.

- **Blast radius.** Code: `structural_harness.gd` (change 0) and `dataset_harness.gd::build_scene` (changes 1, 3) — nothing else. The M0 game scene is **not** involved: the seed is recorded from a synthetic `SynthWorld` with `structrender_no_sim` set, not from Arena Collector. Re-run required: S1–S7 (shared harness), `test_dataset.sh` (M2 recorder test runs against the changed scene; its assertions are content-agnostic), seed re-record + M2 validation, M3 generation + all four gates, M4 retrain. Untouched: M1 exporter, M2 recorder/validator, M3 teacher/alignment, M4 model/training/export code — Decision 37 already established the methodology is sound; only data regenerates.

- **Consequences to record when the new session exists:**  
  - Decision 32's calibration distribution and its "0.013 of headroom" are **scene-bound** and must be re-stated against the new session even if every threshold holds unchanged.
  - Decision 34's measured translation-detection floors (3 px diagonal / 8 px x-only) are scene-bound and must be **re-measured**, with the three docstrings that quote them updated. More vertical structure should *lower* the floor — a genuine improvement to the §12 guard — but that must be shown, not assumed.
  - **§11 ladder comparability — M4 is superseded, not invalidated.** Milestone 4 was legitimately passed on its own terms: its four exit criteria were verified against the data that existed, the thresholds were fixed before the runs, the split was leak-free, and the ONNX path was proven. Nothing about commit `85ca946` was wrong when it was made, and it should not be read as an error. What changes is only *comparability*: once the scene changes, M4's numbers and every do-nothing anchor (native LPIPS 0.0425 / SSIM 0.98988 / PSNR 35.03 dB) describe a scene that no longer exists, so they become **historical reference** rather than the ladder's Model A. Every ladder model trains on the new session, and no cross-scene comparison is ever quoted. The retrain is a re-baselining, not a repair. The sequence-name holdout split (`configs/training/m4_unet_baseline.json`) survives re-recording unchanged, so the split policy itself carries over intact.
  - `ds_1788624547_cf0ceeda` is retired on the same terms as `ds_1788430292_4c77250c` (Decision 34), and its surviving `gouache` target set and `ALIGNED` marker are invalidated with it — they are keyed to that session's frames.
  - Decision 37 asked for "more varied object types and silhouettes"; this scope delivers **boxes** at varied depths, sizes and orientations. That narrowing is deliberate under Simplicity First and is recorded so a second failure of the style sweep is diagnosable rather than mysterious.
  - **As shipped: 13 `add_prop` calls, which is item 1's "6–8" plus item 3, not a scope overrun.** Six are the freestanding mid-depth props item 1 asks for (four in the home frustum, two in the `+Z` group visible only under the 165° aimed-away pose); the other seven are the floor and backdrop restructured into stepped and tilted sections under item 3, replacing the two flat planes that were there before rather than adding to them. Recorded because "6–8 props" and "13 props" otherwise read as a silent expansion of a scope this Decision exists to bound.

---

### Decision 39: The Enriched Scene Rescues the Config That Failed — Decision 37's Diagnosis Confirmed (2026-09-06)

- **The clean experiment.** Decision 37 concluded, from the *absence* of any workable configuration, that scene geometry rather than the teacher pipeline was what limited style strength. That was an inference from failure. It has now been tested directly.

  The oil-impasto config at `strength 0.50 / guidance 9.0 / controlnet 1.4` — **identical prompt, parameters and seed** — was re-run against the enriched scene:

  | | old scene `ds_1788624547_cf0ceeda` | enriched scene `ds_1788687350_337657ea` |
  |---|---|---|
  | oil-impasto @ g9.0 | **FAILED 34 of 990 frames** | **0 of 81 failures**, including all 29 previously-failing frames |
  | worst spurious-edge ratio | 0.930 | 0.159 (gate 0.35) |

  Only the scene changed. This is the direct confirmation of Decision 37, recorded separately rather than folded into a style choice, because a diagnosis that was reached by elimination deserves the positive result written down where a later reader can find it.

- **A second, unanticipated effect: the enrichment lifted every configuration, including the incumbent.**  
  `gouache` — unchanged in prompt, parameters and seed — went from ~0.0512 style signal on the old scene to **0.0907** on the new one. Nothing about the style moved; this is the Decision 38 albedo fix letting authored colour reach the native render for the first time. So the old scene suppressed style signal in **two independent ways**: too little geometry for the conditioning to hold (Decision 37), and colour that never rendered (Decision 38). Decision 37 identified one of them; the second only became visible while scoping the fix for the first.

- **Measured effect of the enrichment, on the previously-failing spans only** (Decision 38 requires this comparison be restricted, since aggregate improvement is guaranteed by adding geometry and proves nothing):

  | `reap_entity_away` gap spans | old | new |
  |---|---|---|
  | reference edges (median) | 834 | **1618** |
  | entity pixels (median) | 1481 | 1481 |

  Structure nearly doubled exactly where it was missing, with entity content unchanged — the props are doing the work, not extra entities, which is what the scope intended.

- **⚠ Unreconciled: the regression set is quoted as 29 frames, but Decision 38's rule implies at least 34.** Decision 38 defines it as "every frame that failed either previous run, carried forward permanently"; Decision 37's table records those runs as failing **34** and **10** frames, whose union is ≥ 34. Decision 39's "all 29 previously-failing frames" and the "29 regression" line below therefore either omit ≥ 5 frames the rule requires, or one of 34 / 29 is a miscount. **This cannot be settled from the repository** — the retired session's target sets and `ALIGNED` marker were deleted with it (Decision 38), so the failing-frame lists no longer exist. Recorded as open rather than silently reconciled, because Decision 39 is the only controlled evidence Decision 40 now rests on and the strength of its "including all previously-failing frames" claim depends on which number is right. If M5 re-records, rebuild the regression set from the run that produces it rather than carrying 29 forward.
- **Standing rule unchanged.** The sweep above is a *stratified subset* (81 frames: 29 regression — see the caveat above + the poorest decile exhaustively + contiguous blocks). It ranks candidates; it does not license pinning. No style is pinned until it passes the full 990-frame gate run on the new session. Decision 37's method note stands: uniform sampling cannot find clustered failure, and only full-set validation is evidence of a pass.

---

### Decision 40: `oil_impasto` Is the Committed Style; `gouache` Retired (2026-09-06)

- **Pinned:** `oil_impasto` — SD 1.5 + ControlNet-depth, strength 0.50 / guidance 7.5 / controlnet 1.6, seed 20260906 — validated **990/990, zero failures** on session `ds_1788687350_337657ea`:

  | gate | threshold | enriched scene (`oil_impasto`) | retired scene (`gouache`) |
  |---|---|---|---|
  | `boundary_f` | ≥ 0.60 | **min 0.848** | min 0.709 |
  | `spurious_edge_ratio` | ≤ 0.35 | **max 0.150** (0.200 headroom) | max 0.337 (**0.013** headroom) |
  | `min_silhouette_support` | ≥ 0.60 | **min 0.781** (n=695, not 990) | min 0.661 |
  | `stylisation_hist` | ≥ 0.15 | min 0.287 | min 0.232 |

  > **⚠ This table is a confounded comparison and is not evidence of the enrichment's effect.** Both columns change *two* things at once — scene **and** style — and Decision 38's measurement discipline is explicit that aggregates over the full session prove nothing here, because adding always-visible props inflates `n_R` with easy edges by construction. The table records **what the committed style scores**, which is its only job. The controlled evidence is in **Decision 39**, and only there: `gouache` held identical in prompt, parameters and seed rose **0.0512 → 0.0907** style signal on the new scene (scene isolated, style fixed), and on the previously-failing spans alone the median reference-edge count went **834 → 1618** with entity pixels unchanged at 1481. An earlier version of this bullet asserted "~2.3× the style signal" against no baseline at all; that figure is withdrawn — no committed artifact reproduces it, and any cross-column ratio here measures the style change as much as the scene change.

  **The margin is confounded too — this is a correction to the first version of this bullet.** It claimed gate 2's worst-frame headroom (0.013 → 0.200) was the one figure here immune to dilution. That is wrong, and **Decision 43 disproves it two entries later**: `spurious_edge_ratio`'s denominator is `n_Es`, the target image's *strong-edge* count (`metrics.py`), and the props are visible in the target image, so they raise it directly — Decision 43 measures 200–700 → **~1800** strong edges per frame. A denominator up ~3× against a worst-frame ratio down 2.25× (0.337 → 0.150) is dilution, not headroom, and Decision 43 already records the same mechanism as a **regression** in localised-hallucination sensitivity. **No cross-column figure in this table is evidence of anything**; the controlled evidence is Decision 39's and only Decision 39's — and specifically its **`gouache`-held-fixed** measurement (0.0512 → 0.0907, identical prompt, parameters and seed, scene the only variable). Decision 39's *oil-impasto* row is a different parameterisation from the one pinned here (it tests `g9.0 / c1.4`; this config ships `g7.5 / c1.6`), so it evidences "the enriched scene rescues a config the old scene failed" — which is what Decision 39 claims — and not anything about the committed style's own numbers.

  **What survives is only the retired-scene half, and a second correction is needed here.** The first version of this bullet claimed a quantitative margin gain; withdrawn above. The replacement then claimed that signal and margin "moved in strict opposition on the retired scene, and on the enriched scene they do not" — the first clause is supported (Decision 37's table: every config with more signal than `gouache` failed, and the only comfortable pass restyled *less*), **the second is not, and the very next bullet of this entry disproves it**: `cel_bold` carried the highest signal of the sweep and failed **36 of 990 on the enriched scene**, for exactly the stated reason — more signal, more strong edges, a fraction unsupported. Decision 43 reaffirms the gate-2 floor is style-dependent, not only scene-dependent.
  So: **no enriched-scene measurement pairing style signal with gate-2 margin exists**, in this log or in any artifact, and none is claimed. What the enrichment demonstrably changed is that a config which previously failed now passes with the scene as the only variable (Decision 39). Whether the signal/margin *tension* is weaker on the new scene is untested — `cel_bold` says it is still there. Recorded because this is the third attempt at this sentence and each earlier one asserted more than the data carries.

- **`gouache` is retired**, not kept as a second style: config deleted, target set deleted, all references repointed. Decision 31's "one real style, every path style-keyed" scope is therefore unchanged — this is a replacement. **This is explicitly NOT the deferred style-B.** Style-B remains an open decision for **M8**, to be chosen deliberately and probably in sharp visual contrast to `oil_impasto`, at the point where EPAM's style-reset behaviour (Decision 12, §11 Scenario 6) is actually under test rather than hypothetical. Retiring a style the ladder outgrew is not the same as choosing the one M8 needs.

- **Rejected candidate, recorded because the failure is instructive:** `cel_bold` (s0.55 / g9.0 / c1.5) carried the highest signal of the sweep and passed a **stratified** 81-frame probe 0/81 — then failed **36 of 990**. Its failures were spread across the whole session rather than clustered, because a stronger style emits 2500–2900 strong edges per frame instead of 200–700 and a fraction land unsupported everywhere. **The probe had been stratified against the *previous* failure mode** (empty frames), which no longer applied. `oil_impasto` was then selected on those 36 empirically-hard frames (0/36, worst spurious 0.063) rather than on any sample.

- **Method note, now demonstrated four times:** across every style attempt in this sequence, a subset probe has **never once** predicted the full-990 result — uniform sampling missed a clustered failure, and stratified sampling missed a distributed one. A subset probe can only test hypotheses you already hold; full-set validation is the only thing that finds the failure you did not predict. Subset probes rank candidates. They are never evidence of a pass.

---

### Decision 41: M4 Adds the Core Perceptual Loss; Detection Floors Re-measured (2026-09-06)

- **Context.** Retraining M4 on the enriched scene and the new style made it fail `test_output_is_not_blurrier_than_the_target`: the L1-only model retained **0.751** of the target's high-frequency energy against a 0.9 floor (native anchor 0.993). This was predicted, not surprising — Decision 35 recorded "Loss at M4 is `L_rec` (L1) alone" as an obligation to revisit, and L1's textbook failure is regression to the blurry conditional mean. It stayed invisible only because the retired `gouache` target had almost no high-frequency content to lose.

- **Chosen Path: `L_rec = 1.0·L1 + 0.5·LPIPS(alex)`** (`renderer/losses/perceptual.py`). This is **not** scope creep past M4's "keep the baseline plain" brief: §4 *Training Loss Module* and **Decision 4** list reconstruction *and perceptual* as **core**, with only the adversarial term optional. The architecture is untouched — still a plain U-Net, no attention, no temporal, no entity machinery, no discriminator. Training **raises** rather than silently falling back to L1 when `lpips` is unavailable, because a ladder model trained under a quietly different objective is precisely the confound Decision 35 exists to prevent.

- **Result — one metric improved, every distortion metric regressed. The ledger, not a summary:**

  ⚠ The **L1-only column is an uncommitted measurement**: it comes from a diagnostic run on the enriched scene whose report was not kept, and no artifact in `eval/reports/` reproduces it. The `L1 + perceptual` column is `eval/reports/m4_metrics.json`. Cited here because the comparison is the whole point of the entry, flagged because Decision 35's "numbers are the ones in the committed artifacts" rule does not currently cover half this table.

  | | do-nothing (native) | L1 only | L1 + perceptual | change |
  |---|---|---|---|---|
  | **sharpness ratio** (gate 0.90) | **0.993** | 0.751 ❌ | **0.930** ✅ | **improved**, +0.179 — and the only reason the suite passes, but see below: it is still **below its own do-nothing anchor** |
  | SSIM gap closed vs do-nothing | 0 % by construction (anchor 0.9640) | 23.7 % | **8.7 %** | **regressed**, lost ~63 % of the gap it had closed |
  | PSNR over do-nothing | 0 dB by construction (anchor 29.95 dB) | +3.53 dB | **+3.33 dB** | **regressed**, −0.20 dB |
  | selection-set L1 | *not reported* — no native-L1 anchor in the artifact | 0.00944 | **0.01006** | **regressed**, +6.6 % |
  | LPIPS gap closed vs do-nothing | 0 % by construction (anchor 0.1010) | 15.2 % | 47.5 % | improved — but **this is the training objective**, so it is not evidence |

  (Three rows — SSIM, PSNR, LPIPS — are already *stated as* a distance from the do-nothing anchor, so their anchor column is zero by definition and carries the absolute value instead. **Two are not anchor-relative: sharpness and selection-set L1.** Sharpness is the one that had an anchor available and was published without it, which is the defect Decision 44 records; selection-set L1 has no native anchor in the artifact at all, so its cell says so rather than inventing one.)

  **Read it plainly: three independent distortion metrics got worse (SSIM, PSNR, L1), one independent metric got better (sharpness), and the one that improved most is the objective itself and therefore proves nothing.**

  **And the one that got better is still worse than doing nothing.** The unstyled engine frame carries **0.9931** of the target's high-frequency energy on this scene; the trained model carries **0.9303** (`m4_metrics.json`, `sharpness_ratio_native` vs `sharpness_ratio`). Every other row above is stated against its do-nothing anchor and this one was not, which flattered it — corrected here. The consequence is a limit on what the gate certifies: **at a 0.90 floor the do-nothing frame passes trivially, so `sharpness_ratio` is a blur *guard*, never evidence of quality.** It is only meaningful paired with gate 4's histogram travel, which the native frame fails outright (Decision 32, Finding 1) — sharp-but-unstyled and styled-but-blurred are two different failures and it takes both metrics to exclude both. The perceptual term moved the model from clearly blurrier than native (0.751) to slightly blurrier than native (0.930); it did not make it sharper than native, and no number here claims a renderer that out-resolves its own engine input. Calling this "a split" would overstate the case — the honest description is that the model traded pixel accuracy for perceptual quality, which is the textbook perception–distortion trade-off, and the trade is only worth making because *sharpness was the gate actually failing* and a blurry renderer is a worse renderer than a slightly less pixel-accurate one. Net assessment: **accept**, on the grounds that M4 is a renderer baseline and not a compression benchmark — but the regressions are real and are carried into the ladder, not written off.

  **Two further cautions:**
  - The surviving margin is thin. Sharpness is **0.9303 against a 0.90 floor — about 0.03 of headroom**. The comparison that matters is the *same-scene* do-nothing **anchor** of 0.9931 above. Not the retired scene's 0.9926 — which was that scene's **model** ratio, not an anchor, under a different scene, style and objective, and Decision 38 forbids the cross-scene read either way. The one independent gate carrying this verdict sits between its floor and its own anchor with little room either way; a busier style would likely cross it.
  - **The coupling is ladder-wide, not M4-specific.** Decision 41 mandates this loss for *every* §11 ladder model, so the ladder's LPIPS column is an objective rather than a measurement **for all rungs**. §11's independent set is therefore **SSIM, PSNR, the sharpness ratio and the four structural gates** — quote LPIPS alongside them, never alone, and always with its do-nothing anchor (Decision 36).

- **Translation-detection floors re-measured** (required by Decision 38; they were scene-bound and must never be inherited silently): **2 px diagonal / 5 px x-only**, improved from 3 px / 8 px on the retired scene. The x-only floor was the weak one because that scene's dominant structure was a long horizontal horizon, near-invariant under horizontal translation; the enriched scene's vertical structure closed the gap. A systematic sub-5 px horizontal misalignment still passes every gate, so the §12 limitation is **reduced, not eliminated**. Updated in all three places that quote it (`metrics.py`, `teacher_targets.md`, `test_alignment_rejects.py`).

- **Model A's ladder numbers** (session `ds_1788687350_337657ea`, style `oil_impasto`, contiguous-sequence split unchanged): L1 0.01006, PSNR 33.28 dB, SSIM 0.9671, LPIPS 0.0531, sharpness 0.930. Every one is a **selection-set** figure under Decision 35's labelling, and every §11 ladder model must use this same loss and selection rule.

---

### Decision 42: Overfit Gate Uses the Configured Objective; Reports Must Self-Identify (2026-09-06)

- **The overfit gate trains under the configured loss**, not a hardcoded `F.l1_loss`. It is a capacity check on the setup actually shipped; running it under a different objective than the ladder would show the model can memorise something it is never asked to memorise. Re-run on the current session and objective — **N = 32, §11's mandated floor of 32–128** — **all four pre-registered thresholds clear with margin: PSNR ≥ 30, final L1 ≤ 0.02, loss drop ≥ 10×, SSIM ≥ 0.95.** Deliberately no measured values here. **The overfit loop drifts between runs** — measurably, unlike training, whose run-to-run behaviour has never been measured at all (an earlier version of this sentence called training "seed-deterministic on this hardware and reproduces the ladder numbers exactly"; that is unestablished, and Decision 35 records why — no deterministic flags are set, and training uses bf16 autocast, worker processes and shuffling). The asymmetry claimed here is only that the overfit loop's drift **is quantified** and training's is not; it is not a claim that training does not drift. and it drifts unevenly: across three N = 32 runs PSNR moved ~1.4 % and SSIM ~0.05 %, but final L1 moved ~5.6 % and the loss-drop factor ~5.9 % (spread relative to the mean). The config's `threshold_notes` states the same quantity as a rounded absolute band, "~53–57×", whose own spread is ~7.3 % — the two are the same three runs described at different precision, not two independent measurements, and they are not expected to agree to the decimal.. Typing its numbers into this entry produced two stale citations in two consecutive attempts, so the rule is now structural: **a quantity that drifts run-to-run is cited by threshold and artifact, never by value.** Read `eval/reports/m4_overfit.json`, which self-identifies its session, style and objective. **Directionally, the current run is worse than the retired scene's on all four** — PSNR, SSIM and the loss-drop factor fall and final L1 rises — because the task is genuinely harder: a textured target under a perceptual objective. Stated as directions and not as numbers on purpose; the retired figures live in Decision 35's ⚠ bullet, which instructs that they not be quoted, and an earlier version of this very paragraph quoted them anyway while stating the rule. Thresholds were never re-tuned across the scene change, the style change or the loss change, and the artifact-level check that the committed report still clears them is `test_committed_overfit_artifact_meets_the_mandated_gate`.

- **Every committed report names the data and objective it came from** — `session_id`, `style_id`, `objective`, `generated_utc`. Their absence is exactly how a *retired-scene* overfit artifact survived undetected and left M4 exit criterion 2 unevidenced: the file's numbers were identical to a run against a scene that no longer existed, and nothing in it could contradict that.

- **Derived, never quoted.** The checkpoint-selection note is now computed from the run's own history. A hardcoded figure published the retired scene's 1.3 % selection gain into a new session's report, understating the real **3.2 %** (best epoch 39 val_l1 0.010064 vs final epoch 59 0.010385) by ~2.5×. That was the **third** stale-citation defect in this project (after Decision 35's overfit numbers and Decision 32's distributions), so the durable fix is derivation, not vigilance.

- **The training history names the objective it logs.** `train_l1` accumulated the *total* loss once a perceptual term existed, so the report claimed an L1 it never measured (0.01013 logged vs 0.005439 actually measured). Renamed `train_loss`, with a `train_objective` field alongside. A ladder curve compared against M5's would otherwise have compared different quantities.

- **`ImportError` only.** `PerceptualReconstructionLoss` caught bare `Exception` when probing for `lpips`, which would report a genuine LPIPS failure (bad weights, CUDA fault) as "unavailable" and silently train under a different objective — the confound Decision 35 exists to prevent.

---

### Decision 43: The Scene Enrichment Traded Hallucination Sensitivity for Translation Sensitivity (2026-09-06)

- **It was not a pure win, and the regression must not be discovered later.** The enriched scene emits **~1800 strong target edges per frame** against the retired scene's 200–700. Gate 2 is a *ratio* — unsupported strong edges over total strong edges — so a fixed-size hallucination is now a smaller fraction of a larger denominator.

- **Measured, on the enriched scene — and the floor is STYLE-dependent, not merely scene-dependent.** Gate 2's denominator is the target's *total* strong-edge count, so a busier style shrinks the same hallucination's share:

  | target style | clean strong edges / frame | detection floor |
  |---|---|---|
  | `oil_impasto` (the committed teacher) | ~1 800 | **6.5 %** of frame (68 × 140, caught 6/6; 44 × 84 caught only 2/6) |
  | `ci_posterize` (the CI teacher) | ~3 234 | **11.0 %** of frame (90 × 180, caught 6/6; 68 × 140 caught **0/6**) |

  A localised hallucination below the floor passes gate 2. On the retired scene a 52 × 120 object was reliably caught, so this is a genuine **regression in localised-hallucination sensitivity** — traded for the improved translation sensitivity of Decision 41 (3 px → 2 px diagonal, 8 px → 5 px x-only).

  **Practical consequence for tests:** `test_hallucinated_object_is_rejected` runs against `ci_posterize` fixtures (no model download), so it must be sized from *that* style's floor. Sizing it from the `oil_impasto` measurement made it fail — the object was above the real teacher's floor and below the CI teacher's. A detection floor is only meaningful paired with the target set it was measured on.

- **Net assessment: accept the trade, record both halves.** Translation error is the failure mode that actually threatened the dataset (a systematically shifted target set corrupts every frame); a sub-6 %-of-frame invented object is a narrower fault. But §12's guard is now *weaker* in one axis than it was, and that belongs in the log rather than in the gap between two green test suites. `HALLUCINATION_FLOOR_HW` in `test_alignment_rejects.py` pins the floor so a further regression fails rather than passing quietly.

- **Consequence for gate 2's threshold.** Do **not** lower `spurious_ratio` to compensate: the observed max on 990 clean frames is 0.150 against the 0.35 gate, so tightening would still not catch a 2.5 % object (which scores 0.295–0.428 depending on frame) while risking false positives on legitimate painterly strokes. If localised-hallucination sensitivity matters more later, the correct fix is a **spatially-local** gate — largest connected unsupported-edge region, rather than a whole-frame ratio — which is a new metric and needs its own entry.

---

### Decision 44: Pre-Commit Review — A Pre-Registered Threshold Changed Units; Two Guards Did Not Guard (2026-09-08)

Recorded from the second M4 spec-guardian pass, which was run because the first pass's findings had largely been self-corrected and a clean review was wanted rather than a shorter list. It was not clean. Three things here are substantive; the rest of the pass's findings were stale citations corrected in place in Decisions 34, 35, 38, 40, 41 and 42, each marked where it was corrected. A **second** verification pass over those corrections then found four more, three of them introduced by the corrections themselves — a broken table column, a false margin claim this log disproves two entries later, and this entry describing its own new test backwards. Those are fixed in place too and are the reason this bullet list is longer than the defects warranted.

- **`loss_drop_factor` silently changed units when the objective changed, and no entry recorded it.** It is `step0_loss / final_l1`. When M4's loss *was* L1 that was a like-for-like ratio, and Decision 35 pre-registered it as "final L1 at least 10× below the step-0 loss". Decision 41 made the objective `L1 + 0.5·LPIPS`, so the numerator is now the **total objective** and the denominator is still pure L1 — a **mixed-unit ratio**. The code says so (`objective_note`, written into the overfit report) and the config now says so too (`overfit.threshold_notes` — a different key, named here because the entry whose subject is wrong citations should not contain one); the Decision Log did not, and a pre-registered threshold whose definition moves needs its own dated entry rather than a footnote in an artifact.
  **It stays as a sanity check and is not repaired.** Decision 35 already downgraded it: the head is sigmoid-initialised against dark targets, so step-0 loss is large by construction and clearing 10× is close to vacuous — it measures ~53–57× on the current session under the perceptual objective — the config's `threshold_notes` carries that range and the artifact carries the run's own value. The retired scene's range was higher; it is **not repeated here**, because Decision 35's ⚠ bullet that holds it says not to quote its numbers and Decision 42 records a previous paragraph doing so while stating the rule. Each of the three places that mention a range is scoped to its own session, and none is comparable to another. Making it like-for-like would mean logging a separate step-0 L1 purely to feed a check that carries no verdict. **PSNR, final L1 and SSIM are the three gates that carry the overfit verdict**; this is the fourth number in the report and is explicitly not one of them. What was wrong was leaving that undocumented, not the ratio itself.

- **Decision 38's stated occlusion guard did not exist, and the assertion added for it was a duplicate.** Decision 38 said a stray full occlusion "creates a second hidden→visible transition and the M2 validator rejects the session". The validator does not: `_check_sequences` finds the **first** hidden stretch and checks only that one is clean. The Godot-side assertion added alongside it (`_check_gap_is_contiguous`) compared the derived gap against the configured gap — which is character-for-character the check already three lines above it in the rung loop. So the rule Decision 38 required to be "asserted, not eyeballed" had two layers of coverage for the case that moves a gap boundary and **none** for the case that does not: an occlusion sitting entirely inside the POST margin. Replaced with `_check_one_hidden_run`, which walks the target's visibility across the rung's own `frame_range` and requires exactly one visible→hidden transition, with two non-vacuity guards (the target must be present in `entities[]` for every frame of the range, and the derived first-hidden frame must equal `last_seen_frame + 1`) — without those, a failed visibility parse would report exactly one run and pass green while measuring nothing. Decision 38's sentence is corrected in place, in the KB and in both GDScript comments that repeated it.
  **Scope limit, stated rather than left to be discovered.** This runs in the CI fixture (`d1_recorder.gd`), which exercises `camera_away` rungs only. The 990-frame seed is recorded by `seed_record.gd`, which runs **both** mechanisms and performs no visibility check — and per the finding above the M2 validator will not catch a second hidden run there either. The geometric risk is low (`entity_away` teleports the target to `(1000, 0, -6)`, so it never traverses the props) but it is not zero and it is not asserted. Closing it means either running the check over the recorded seed or moving it into the validator; deferred, and recorded here so "the occlusion rule is asserted" is not read as covering the artifact that actually becomes training data.

- **Nothing asserted that the *committed* overfit artifact passed.** `test_overfit.py` asserts against a freshly computed result and skips entirely without CUDA and ≥ 4 GB VRAM; `test_report_provenance.py` checked who produced a report, never what it said. A committed `m4_overfit.json` recording `n_frames: 8` — **the exact §11 violation this project already shipped once** — passed the whole suite on any machine without a GPU. `test_committed_overfit_artifact_meets_the_mandated_gate` now reads N **from the artifact** and cross-checks it against the config, reads the four thresholds from the config and requires the artifact to have been scored against exactly those, and holds §11's 32–128 range as a literal — correctly, since it is a KB constant and not a tunable. Residual, now stated in the test's own docstring: a *simultaneous* config-and-artifact relaxation still passes — lower a threshold in the config and rescore the artifact against it and every assertion holds — so a pre-registered threshold is only pre-registered if someone reads the config diff. The docstring previously claimed the opposite ("relaxing either shows up as a config diff"), which was true of N and false of the thresholds. Verified by re-injecting both defect shapes (N = 8; a relaxed `min_psnr_db`): both caught, artifact restored byte-identical.

- **Sharpness was the row in Decision 41's ledger that had a do-nothing anchor available and was published without it, and the omission flattered it.** (Selection-set L1 also carries no anchor, but none exists in the artifact to carry — a different situation, and its cell now says so.) Corrected there: the native frame scores **0.9931**, the model **0.9303**, so on the single independent gate that carries Decision 41's verdict the model is **blurrier than doing nothing**. The gate is still met (floor 0.90) and the assessment still stands, because `sharpness_ratio` is a blur guard that the native frame passes trivially while failing gate 4 outright — but a ledger that states every other metric against its anchor and omits this one is the softened summary Decision 41 was written to avoid.

- **Asymmetry to record: the M3 teacher-gate figures are prose-only.** `eval/reports/` holds the four M4 artifacts and nothing else. Decision 40's gate table, Decision 43's strong-edge counts and detection floors, Decision 39's 834 → 1618 and 0.0512 → 0.0907, and `metrics.py`'s four distributions have **no committed artifact** — the `ALIGNED` marker and target sets are gitignored, and the retired session's were deleted outright. Decision 41 was careful to ⚠-flag its own uncommitted L1-only column; nothing equivalent flags these, and the method note below is therefore not true of them. Only `HALLUCINATION_FLOOR_HW` is pinned by a test. This is not a defect in any single number — teacher generation is seed-fixed and reproducible — but "cited by artifact" is currently unavailable for the entire M3 layer, and the honest options are to commit an alignment report alongside the M4 ones at the next generation run, or to record the exemption explicitly. Deferred to M5's re-generation, flagged here so it is not mistaken for coverage that exists.

- **Method note.** Of the eight deviations this pass found, five were stale or unattributed *numbers* in prose — the seventh through eleventh such defects in this project. Decision 42's structural rule ("cited by threshold and artifact, never by value") was stated in the same paragraph that then cited a drifting value. The rule is right and vigilance is again not the mechanism: the citations that survived this pass are the ones a test or a derivation produces. Where prose must carry a number, it now carries the marker saying which scene it describes.

### Decision 45: M5 Export Probe — §12's Attention Risk Did Not Materialise; Four Written Rules Did Not Survive Contact (2026-09-08)

The M5 skeleton export probe, run **before** the architecture was built, per §12's "perform TensorRT export in Week 5" and to keep an export blocker from surfacing after a trained model. Evidence: `eval/reports/m5_export_probe.json`, generated only by `scripts/probe_m5_export.py`. Every engine below was **built and executed**, not merely exported.

- **The headline: §12's flagged risk — "custom attention/masking operation exports poorly" — did not materialise.** 19/19 variants export to ONNX opset 17, build and run under TensorRT in both FP32 and FP16, keep fully static shapes, and land inside M4's 1e-3 parity tolerance. Cited by threshold and artifact, per Decision 42; the per-variant figures are in the report. This closes §12's export risk for self-attention, window partitioning, relative-position bias and additive masking, and **opens** nothing new — the M7 entity-masked cross-attention probe passed on the same terms.

- **Attention is manual `matmul → +bias → softmax → matmul`, not `F.scaled_dot_product_attention`.** At opset 17 the exporter emits **no fused attention op** for SDPA. It emits the same matmul/softmax/matmul plus a runtime scale read off the tensor shape (`Shape → Slice → Cast → Sqrt → Div`), which `do_constant_folding` cannot fold because the input is a `Shape` of a graph tensor. Measured on the end-to-end graph: `Cast ×12` and `Sqrt ×18` appear **only** in the SDPA graph, 574 vs 508 ONNX nodes, 1519 vs 1291 TensorRT layers, for numerics asserted bit-comparable. SDPA's entire justification is a fused kernel; it does not survive the export boundary, so it costs a per-block shape read and buys nothing. Kept behind an `attn_impl` flag so this is re-testable rather than folded away — if a later opset emits a real fused op, the comparison reruns.

- **`torch.roll` is not the hazard §12 anticipated.** ONNX has no `Roll`, so the exporter lowers it to exactly the slice+concat pattern one would hand-write: roll and an explicit slice+concat build **identical TensorRT engines** (311 layers, both), differing only by 8 `Constant` nodes in the ONNX census. Shifted windows are therefore kept — §4 requires "broader object-level interaction than a shallow CNN", and without shifting the window boundaries are hard walls — and the workaround exists, is verified, and is not needed.

- **The probe earned its keep on a defect nothing else would have caught.** The *fallback* slice+concat implementation computed `shift % x.shape[1]`, reading a shape at runtime and leaving the exported output dims **symbolic** — a violation of §12's "prefer fixed/bounded shapes" that TensorRT **still built successfully**, so it would have shipped silently and been discovered, if ever, as a runtime shape failure. `shapes_are_static` is now asserted per variant in the report and in the tests. A green engine build is not evidence of a static graph.

**Four written rules that did not survive contact with the implementation. Each is a deviation from §4/§5 as written, recorded rather than done quietly.**

1. **TensorRT 11 removed `BuilderFlag.FP16` / `INT8` / `BF16`.** §4 *TensorRT Builder* describes "primary precision: FP16" as a builder setting. On TensorRT 11.2 it is not: reduced precision comes **only** from a `NetworkDefinitionCreationFlag.STRONGLY_TYPED` network that inherits dtypes from the ONNX graph, so an FP16 engine requires exporting an **FP16 ONNX graph** (traced on CUDA — LayerNorm/GELU have no CPU fp16 kernels). This changes M10's export-module contract: it must be able to emit an FP16 graph, not merely ask the builder for one. 19/19 FP16 engines built and ran under this approach.

2. **§4's `A_{p,i} = -\infty` for `p ∉ S_i` is literally unusable, in two independent ways — both verified, not argued.** A literal `-inf` makes softmax return **NaN** for any pixel belonging to no entity, which is most of a frame. And because softmax normalises to 1, a *finite* large-negative mask alone still makes a fully-masked background pixel emit a **blend of every entity's memory** (measured: attention row sums to 1.0 over illegal keys). The implementation uses a finite `MASK_NEG` **plus** an output gate `max_i S_i(p)`, which drives background pixels to exactly 0.0. This is read as the **intent** of §4's rule rather than a relaxation of it — "no legal key" must mean "no contribution", which `-inf` alone does not deliver — and it is asserted by tests: perturbing entity *i*'s token provably moves nothing outside `S_i`. Recorded because the formula in §4 stays as written and the code deliberately differs from it.

3. **Style conditioning enters as a second graph input.** The end-to-end M5 graph is `(structure(1,7,288,512), style_token(1,16)) → rgb(1,3,288,512)`. §5's Runtime Tensor Contract already lists a style token, and M5's deliverables include style conditioning, so this is an interpretation rather than an addition — but it makes the exported graph **two-input**, where M4's was one, and M10's runtime must bind both.

4. **Probe-scale substitutions, recorded so they are not mistaken for design.** `K_max = 8` in the probe against §5's recommended 32→64 (`k_max_probe` in the report — probe scaling only, not a change to Decision 21's ranking rule); and encoder downsampling uses a stride-2 `Conv` rather than Swin patch-merging, chosen for ONNX safety.

- **Scope, stated so this entry is not over-read.** Nothing was trained. No M5 exit criterion is claimed. The probe ran at embed dim 32, 1 head, 1–2 blocks, so its latency figures rank variants **against each other** and are **not** evidence about §12's 15–20 ms StructFormer budget, which depends on trained width. The report carries this as `latency_note`. Width and depth are deliberately unanswered here.

- **New operators versus M4's exported graph** (§12 wants a new op to be a visible diff): `Cast, ConstantOfShape, Div, Erf, Expand, Gather, Gemm, Identity, LayerNormalization, MatMul, ReduceMax, Slice, Softmax, Sqrt, Sub, Tile, Transpose, Unsqueeze`. All TensorRT-supported. `Erf` is GELU (opset 17 has no `Gelu`; it arrives at 20). `Cast`/`Sqrt`/`Div` are SDPA-only and disappear under the manual choice above. `ReduceMax`/`Sub` are entity-masking only. Enforced per-variant against the committed report.

---

### Decision 46: Training Run-to-Run Variance Measured — M4's Sharpness Gate Is Seed-Fragile (2026-09-08)

Measured to pre-register M5's better/competitive margin **before** StructFormer trains, per the same discipline that fixed M4's overfit thresholds in advance. Evidence: `eval/reports/m4_seed_variance.json` — three Model A runs identical to `configs/training/m4_unet_baseline.json` except `train.seed` and `train.checkpoint_dir` (split, loss, data, overfit and style_acquisition blocks byte-identical).

- **This replaces a deleted claim with a number.** Decisions 35 and 42 both asserted training was "seed-deterministic" and "reproduces the ladder numbers exactly"; Decision 44 removed both as unestablished. Removing them was right, but an earlier version of THIS bullet then claimed they were "also wrong: training does not reproduce exactly" — a **category error**, corrected here. All three runs use **different seeds**, and different seeds are expected to differ; that says nothing about whether the *same* seed reproduces. **Same-seed reproducibility remains untested.** What the three runs measure is across-seed spread, and only for the CNN. Per-metric figures are in the artifact.

- **A two-run estimate understated the noise and would have shipped a margin narrower than the thing it measures.** The first two seeds differed by 0.061 dB PSNR; the third took the sample sd to 0.165 dB — the raw difference is **2.7×** smaller, and an n=2 sample sd (0.043) is **3.8×** smaller. (An earlier version of this bullet said "~5×", which no computation from the committed numbers supports, and claimed a narrower band would let "almost any result be called better" — that is backwards: a narrower band classifies results *more* decisively, and attempt 1 would have been called worse under it too. The defect in an n=2 margin is that it is not a variance at all, not that it flatters.) **n=2 is one difference, not a variance.** Recorded because the cheap version of this measurement is actively misleading rather than merely weak.

- **M4 exit criterion 3(c) is seed-fragile, and this is the substantive finding.** `sharpness_ratio` has the largest relative spread of any metric measured (sd ~0.014 against a **0.90** floor). The worst of the three seeds lands at **0.90842 — under 0.009 of headroom** — so the committed 0.9303 is one draw from a distribution whose gate margin is roughly 2.2 sd, and **a different seed could plausibly have failed the gate.**
  **M4 is not retracted.** The committed run passed, its thresholds were fixed before it ran, and Decision 38 already treats M4 as superseded-but-historical. What changes is how firmly the result may be stated: "sharpness 0.9303 ≥ 0.90 ✅" reads as a property of the architecture and is a property of one run. This also sharpens Decision 41's caution — that entry warned "a busier style would likely cross it"; a different *seed* nearly does.

- **Consequence for every ladder rung, M5 onward: a single run is not evidence against a gate whose margin is within ~2 sd of it.** Where a metric's gate margin is that tight, the verdict needs either repeated seeds or an explicit statement that it rests on one draw. This applies first to StructFormer's own sharpness, which must be reported against the 0.90 floor, the 0.9931 do-nothing anchor (Decision 41), **and** this variance.

- **⚠ What this spread is and is not.** n = **3**, and the runs are **not environmentally uniform**: seed 20260905 ran under `.venv` (torch 2.14.0+cu130), seeds 20260908/20260909 under conda `dl` (torch 2.13.0+cu132). The sd therefore pools seed variation *and* a toolchain change, from three samples. It is a **preliminary estimate of CNN run-to-run spread** — not a confidence interval, not a tolerance, and **not a measurement of StructFormer variance**, which has never been measured and cannot be substituted for. The thresholds below are retained as **historical operational rules**: fixed before the runs, and what those runs were judged against. That is their value; they are not a statistical guarantee.

- **Pre-registered M5 margin**, fixed now and not revisable after seeing StructFormer's numbers. The reference is the **mean of the three Model A runs**, not the single committed run, because that run is now known to be one draw: **better = PSNR at least 2 sd above the mean; competitive = within ±2 sd; worse = below.** Guard metrics (SSIM, sharpness) may not regress by more than 1 sd. The PSNR margin is in `m4_seed_variance.json` -> `m5_margin`; the guard sds are in that file's `statistics` block, **not** in `m5_margin` — an earlier version of this sentence said they were. Deliberately not retyped here, per Decision 42.

- **Incidental, and it removes a confound rather than adding one:** seed 20260908 re-baselines Model A in the conda `dl` env (torch 2.13.0+cu132) against the committed run's `.venv` (torch 2.14.0+cu130). The environment change is therefore *inside* the measured spread rather than sitting alongside it untested, and both ladder rungs are now same-env.

---

### Decision 47: Ladder Rungs Are Compared at Parameter Parity; Gradient Accumulation Keeps Batch Out of the Comparison (2026-09-08)

> **Naming note (Decision 50):** this entry originally called the M5 StructFormer "Model C". It is **Model C**. §10 Milestone 9's table assigns B to the *CNN temporal* rung, which is M6's deliverable. All configs, tests and entries have been relabelled.

- **Model C was first built at 0.160x Model A's parameters under a stated parity rationale.** The specified width (embed 96) measured 1,259,755 against Model A's 7,853,763. The stop condition given to the builder was **one-sided** ("halt if more than ~1.5x"), so a 6x *undershoot* passed unchallenged. Undershoot is the worse direction: it confounds the comparison precisely where a transformer would look bad for a reason that is not its architecture. `test_model_b_capacity_against_model_a` now asserts a **two-sided 0.9-1.1x band**, not a pin plus an upper bound.

- **Parity lands at embed 240 — 7,775,467 parameters, 0.990x — which OOMs at batch 8 on the 8 GB target card.** Running Model C at batch 4 instead would make **batch size a second difference** between two rungs that §11 intends to differ only in architecture. Resolved with gradient accumulation: `batch_size 4 x accum_steps 2`, so the optimiser sees the same 8 samples per update — **for 80 of 81 updates per epoch.** The 81st is a trailing partial group: 161 batches do not divide by 2, so the last update fires on a single batch of 4 samples still scaled by `1/accum`, i.e. a half-magnitude gradient from half the effective batch. Recorded rather than repaired: the committed runs were trained with this behaviour, and changing the loop now would leave the artifacts unreproducible by the code that ships with them. **Its effect on the historical M5-vs-M4 comparison is UNMEASURED.** An earlier version of this bullet said it was "far below the seed variance of Decision 46"; nothing measured it, and the spread in that entry is a CNN quantity that cannot bound a StructFormer training difference. Two asymmetries are involved, not one: the CNN makes **80** optimiser updates per epoch over **640** samples, StructFormer **81** over **644** (`drop_last` with batch 4 vs 8). Corrected for future runs in Decision 51; the committed runs keep the behaviour they were trained with. `accum_steps: 1` is the default and is byte-identically the previous code path, so Model A is untouched.

- **Two things had to be right for that to be true, and one of them was initially wrong.**
  - The loss is scaled by `1/accum` before `backward()`, so the accumulated gradient is the **mean** over the effective batch. Without it the effective learning rate is multiplied by `accum`.
  - `CosineAnnealingLR`'s `T_max` was sized by `len(loader)`, which counts **batches**. With accumulation the scheduler steps once per *optimiser* step, so at `accum = 2` cosine would have annealed only to its **midpoint** and Model C would have trained under a different LR schedule than Model A while appearing to share one. `T_max` now counts optimiser steps.
    > **⚠ Correction.** An earlier version of this bullet claimed "both rungs compute **4860**". They do not. `drop_last=True` makes Model A `644 // 8 = 80` batches/epoch, so **T_max 4800**; Model C is `644 // 4 = 161` batches, `ceil(161 / 2) = 81` optimiser steps, so **T_max 4860** — a residual **1.25 %** difference in schedule length that gradient accumulation does not remove, because it comes from `drop_last` interacting with batch size rather than from the accumulation itself. Confirmed by the artifacts: `m4_train.json` `steps_per_epoch: 80` against `m5_train.json`'s `161` batches. The fix this bullet describes is real and necessary — without it cosine would have annealed only halfway — but it produces *nearly* identical schedules, not identical ones, and nothing in the suite computes `T_max`. The effect is far smaller than the measured seed variance (Decision 46), so it does not change any verdict; the false claim of exact equality is corrected rather than the number defended.
  - `test_the_training_schedule_is_model_a_s_apart_from_the_checkpoint_dir` exempts `batch_size` and `accum_steps` **only** because it now asserts their product equals Model A's. Exempting them without that assertion would have converted the test into a rubber stamp.

---

### Decision 48: M5 Export and TensorRT Benchmark — §12's Risk Landed on the Baseline, Not the Transformer (2026-09-08)

Measured at **trained width** (7.78 M parameters), not the skeleton of Decision 45. Evidence: `eval/reports/m5_onnx_export.json`.

- **M5 exit criteria 3 and 4 are met.** ONNX export succeeds (opset 17, 492 nodes, every shape static), onnxruntime parity is inside M4's 1e-3 tolerance, and a TensorRT engine **builds and runs** in both FP32 and FP16. Latencies are in the artifact. **Scope:** attempt-1 checkpoint, **batch 1, 512x288, TensorRT engine time only** — CUDA events around `execute_async_v3`, excluding host/device transfers, structural-buffer export and Godot's own frame cost, so it is not an end-to-end FPS figure. Against §12's 15-20 ms StructFormer *allocation* inside a 33.3 ms frame, **FP16 is roughly 4x under that allocation and FP32 also fits** — but every other line of that budget is unmeasured, so 30 FPS end-to-end is not demonstrated. p95 was not computed; only mean and p50 exist. §12 calls those figures engineering targets rather than guarantees; this is the first measurement against them.

- **The custom attention needed no workaround whatsoever.** Window partition/reverse, the relative-position-bias gather, additive masking and `torch.roll` shifted windows all exported and built untouched at 26x the probe's width. §12's "custom attention/masking operation exports poorly" did not materialise at either scale.

- **One genuine workaround, confirming Decision 45 item 1 at full scale:** TensorRT 11 removed `BuilderFlag.FP16`, so the FP16 engine required exporting an **FP16 ONNX graph** traced on CUDA (LayerNorm and GELU have no CPU fp16 kernels), fed to a `STRONGLY_TYPED` network.

- **FP16 parity exceeds the 1e-3 tolerance and is recorded as exceeding it, not waived.** The tolerance was written for FP32 against an FP32 reference; half precision carries ~1e-3 relative precision, so the number is expected — but "expected" is not "inside the gate", and **FP16 needs its own tolerance before it is used as an acceptance criterion.** Deferred to M10, where FP16 becomes the deployment target.

- **The unexpected result: Model A cannot build a TensorRT engine at all.** `Error Code 10: Could not find any implementation for ForeignNode[/enc.0/enc.0.0/Conv.../Sigmoid]` — an NVRTC compilation failure inside a Myelin fused region. Reproduced **4/4** across tf32 on/off x workspace 1024/256 MiB. It is **not generic**: a minimal `Conv->ReLU->Conv->Sigmoid` at the same input shape builds, and Model C also ends in sigmoid and builds. Model A's distinguishing operators are `Resize` (interpolate upsampling), `MaxPool` and `InstanceNormalization` — the last being what §4's GroupNorm lowers to, per Decision 35.
  **Cause narrowed, not identified. Recorded as open rather than guessed.**
  **Scope:** this does **not** retroactively fail M4. M4's exit criterion 4 was "export works", verified as ONNX export plus onnxruntime parity, and §10 deliberately places the TensorRT benchmark at M5, where TensorRT was first installed. What it establishes is that **Model A's TensorRT build is UNRESOLVED in this environment** (TensorRT 11.2.1.2, CUDA 13.2, RTX 5060 Laptop) **while the model §12 flagged as the export risk builds and runs**. An earlier version of this bullet said the CNN has "no TensorRT path"; that overstates an unreproduced, undiagnosed failure on a single toolchain. A different TensorRT version, an ONNX-simplifier pass, or a rewrite of the `Resize`/`InstanceNormalization` path are all untested. The inversion of the risk register is real; the CNN being undeployable in general is **not** established.

---

### Decision 49: M5 Negative Result — StructFormer Underperforms the CNN Baseline at Parameter Parity on This Scene (2026-09-08)

**Milestone 5 does not pass.** Three of its four §10 exit criteria are met — stable training, ONNX export, and the TensorRT benchmark (Decision 48). The fourth, "better or competitive output against CNN baseline", **fails on two independent deliberate attempts**. Recorded as a result, not as a problem to tune away: §11's ablation ladder exists to be able to return this answer, and a rung that cannot lose cannot inform anything.

- **Both attempts were scored against a margin fixed before either model existed** (Decision 46, `eval/reports/m4_seed_variance.json`): better at least 2 sd above Model A's 3-seed mean, competitive within ±2 sd, worse below. Neither attempt reached the competitive band. Per-metric figures are in `eval/reports/m5_metrics.json` (attempt 1) and `m5_metrics_warmup.json` (attempt 2); deliberately not retyped here, per Decision 42.

- **Attempt 2 is a second attempt, not a correction — and it made things worse.** The one hypothesis worth testing was that the schedule inherited Model A's LR with **no warmup**, the standard suspect for a soft transformer, and it was flagged as such *before* attempt 1 trained rather than invented after seeing the result. Attempt 2 changed `warmup_epochs` 0 -> 5 and **nothing else** (model, split, loss, data, overfit and style_acquisition blocks byte-identical; verified). It came out **worse than attempt 1 on PSNR, SSIM, L1 and sharpness alike.** The hypothesis is refuted, not merely unconfirmed, and **attempt 1's numbers stand unchanged in the log.**

- **The failure is not marginal on the metric that matters.** Attempt 1 missed the competitive band by less than 0.1 dB PSNR — close enough that eyeballing it would have been tempting, which is what pre-registration is for. But **sharpness is not close**: both attempts fall well below the **0.90** floor, several sd beneath Model A's *worst* seed, and both fall outside a **0.873-0.927** band derived from the CNN's sharpness spread. That band was the operational rule used at the time to decide whether to spend more GPU on repeat seeds, and both attempts sit far outside it. It does **not** license "one run settles the architecture": what was measured is one run of one implementation, at one recipe, on one scene and one style. StructFormer's own run-to-run spread is unmeasured, so the distance from the floor is suggestive, not conclusive. That band is a *derivation* from Decision 46's pre-registered quantities — the 0.90 floor ± 2 sd of the measured sharpness spread — not a literal string written down in advance; Decision 46 states the rule as "a single run is not evidence against a gate whose margin is within ~2 sd of it". Recorded as a derivation so it is not read as pre-registered text it never was. Both attempts fall outside it by a wide margin either way. Both attempts also land **below the do-nothing SSIM anchor** — the unstyled engine frame scores better on structural similarity than the trained transformer, which Model A never did. Both do pass the histogram-travel gate, so StructFormer *is* restyling; it is doing it soft.

- **What this does and does not license concluding.** It is one scene, one style, and **644 training frames** — a small-data regime where a U-Net's convolutional inductive bias is expected to be worth more than attention's flexibility. This result says *StructFormer at parameter parity underperforms the CNN baseline on this dataset*; it does **not** establish that window attention is wrong for the architecture, and it is not measured against the project's actual headline claim, which is **RCS versus time-outside-view** (§11) rather than image quality. **No third tuning attempt is proposed here.** Deciding whether M6-M8 continue on StructFormer, on the CNN, or on a re-scoped dataset is a project-direction call that belongs with the owner, not a knob to turn.

- **The opposite finding on deployment, and it cuts the other way.** Decision 48 records that Model C builds and runs on TensorRT comfortably inside §12's latency budget while **Model A cannot build an engine at all**. So the ladder currently reads: the CNN wins on quality and has no deployment path; the transformer loses on quality and deploys. Both halves belong in any decision about which architecture carries M6 onward, and neither is a reason to suppress the other.

---

### Decision 50: M5 Review — The Ladder Letter Was Wrong and the Export Contract Reversed Itself (2026-09-08)

From the M5 spec-guardian pass. The negative result of Decision 49 survived review unchanged — every figure it cites was re-derived from the artifacts and the pre-registration ordering was verified by timestamp. What did not survive was the record around it: six blocking findings, four of them **claims in the new entries that the committed artifacts contradict**. That is the pattern Decision 44 catalogues, recurring inside the entries that were written to avoid it.

- **The M5 StructFormer is Model C, not Model B, and the mislabel had propagated everywhere.** §10 Milestone 9's table is the KB's only lettered ladder: **A** CNN baseline, **B** CNN *temporal*, **C** StructFormer baseline, D StructFormer temporal, E entity-aware, F full StructRender. M5 delivers a frame-independent StructFormer — **C**. Every M5 config, test, docstring and Decision entry said B, and one test docstring asserted a *false statement about the KB* ("§11 places Model B at Temporal: No"), where §11 in fact contains no lettered ladder at all. **This would have collided head-on at M6**, whose deliverable is the temporal CNN rung — the real Model B. All relabelled; the repeated citation "§11's ablation ladder" is corrected to §10 Milestone 9.

- **The shipped export graph is one-input; Decision 45 item 3 pinned two.** That entry recorded style conditioning entering as a second graph input and instructed M10's runtime to bind both. What actually exports is `StructFormerSingleStyle`, which folds M5's single cached style token into a **learned parameter**, so it never crosses the graph boundary — `eval/reports/m5_onnx_export.json` records one input, `structure`. **The one-input form is correct for M5** (one committed style, and a token that is constant at inference has no business being a runtime input), but it reverses a written rule silently, which is exactly what Decision 45's own "four written rules did not survive contact" section exists to prevent. **The contract, decided here:** M5 exports one input; the two-input contract belongs to the multi-style `StructFormer` class and returns at **M8**, when style-B makes the token genuinely variable. M10's runtime binds what the shipped graph declares, and the artifact now carries an `export_contract` block saying so.

- **`Equal` and `Where` are in the shipped graph and were on no allow-list.** They are the exporter's lowering of `.expand(structure.shape[0], -1)` on the cached style token — a direct consequence of the one-input form above, which is why Decision 45's probe list (exported from the *two*-input class) did not contain them. §12 requires a new operator to be a **visible diff**, so `m5_onnx_export.json` now pins the shipped graph's full operator census and a test asserts the census is a subset of it. Both are TensorRT-supported and neither caused a build problem; the defect was that nothing would have noticed them.

- **Decision 48's artifact had no generator and no test — Decision 44's finding, one milestone later.** Every claim in that entry rested on a hand-placed JSON that nothing produced and nothing asserted. `scripts/export_benchmark_m5.py` is now the only writer, `renderer/structformer/tests/test_m5_export_report.py` asserts the artifact's own contents (static shapes, parity inside tolerance, both engines built **and ran**, the operator allow-list, and that the Model A failure stays recorded as *not identified*), and `test_report_provenance.py` was extended to cover all six M5 reports — none were under provenance or objective-consistency checking at all.

- **Two exactness claims in Decision 47 were false and are corrected in place**, not defended: the two rungs compute T_max **4800 and 4860**, not both 4860 (`drop_last` interacting with batch size, ~1.25 %); and the optimiser sees 8 samples per update for **80 of 81** updates per epoch, the 81st firing on a trailing partial group of 4. Neither effect was measured, and an earlier version of this bullet asserted both were "far below the seed variance of Decision 46" — unsupported, since that spread is a CNN quantity. Both were stated as exact properties, and an entry whose subject is "keep batch out of the comparison" cannot carry an inexact claim about batch. See Decision 51.

- **The artifact now carries what the prose says about FP16.** Decision 48 correctly recorded FP16 parity exceeding the 1e-3 FP32 tolerance without waiving it, but the JSON stage read `"ok": true` with no tolerance field — and this project has standardised on reading artifacts, not prose. The stage now carries `fp32_tolerance_exceeded` and a note, and a test fails if the exceedance is ever silent.

- **Still open, recorded rather than closed:** seed 20260909 — the run that produces the worst-seed sharpness finding *and* takes the sd from 0.043 to 0.165, i.e. the run that determines the entire M5 margin — has **no committed per-run config or metrics artifact**, only its row in `m4_seed_variance.json`. This is Decision 44's "cited by artifact is unavailable for this layer" exemption recurring, and it is declared here rather than discovered later.

---

### Decision 51: M5 Evidence Audit — A Generator That Could Not Reproduce Its Own Report, and Statistics Stated Beyond What Was Measured (2026-09-08)

Cleanup pass over M5's evidence, run before starting another milestone. No training, no threshold changes, no third attempt: **both StructFormer attempts and the recorded negative result of Decision 49 stand unchanged.** Every finding below was verified from code and committed artifacts before anything was edited.

- **`scripts/export_benchmark_m5.py` produced none of four fields the committed report carried.** `operator_allow_list`, `model_a_comparison`, `export_contract` and the FP16 tolerance flags had been **hand-patched into the JSON after the run**. So the artifact validated against itself, the generator could not reproduce it, and this is Decision 44's finding — a committed artifact nothing can regenerate — recurring in the very commit that added Decision 50 about it. The generator now declares `REQUIRED_TOP_LEVEL`, emits every field, and **refuses to write a report that fails its own schema**. `test_report_matches_the_generator_schema` imports those constants, so generator/report drift now fails a test instead of surviving to the next regeneration. Verified by re-injecting two defects: a missing declared field, and an allow-list derived from the observed graph. Both caught.
  **The committed report is not yet fully generator-produced**, and says so in a `provenance_note`. Its measurements are the originals and are unchanged; the added fields are documentation. A regeneration on a CUDA+TensorRT machine is pending and is the moment that claim becomes true.

- **The operator allow-list is now a declaration, not an observation.** It had been written as the exported graph's own census — which can never fail, because it is derived from the thing it validates. It is now `ALLOWED_OPS` in the generator, and a new operator at M6 must fail and be added deliberately, which is what §12 asks for.

- **Model A's TensorRT failure moved to its own dated artifact** (`eval/reports/m4_tensorrt_build_failure.json`), referenced by name rather than embedded. The generator cannot re-measure it, and carrying its numbers inside a freshly dated M5 report would present an old observation as a new one. Its scope is corrected there and in Decision 48: **unresolved in this environment**, not "no deployment path".

- **The gradient-accumulation asymmetry is larger than recorded, and its effect is UNMEASURED.** Verified from the committed configs: the CNN makes **80** optimiser updates per epoch over **640** samples; StructFormer makes **81** over **644**. On top of that the trailing group — one microbatch, not two — was divided by `accum_steps` rather than by its real size, applying a half-magnitude update once per epoch. Earlier entries called this "far below the seed variance of Decision 46". **Nothing measured it**, and that spread is a *CNN* quantity which cannot bound a StructFormer training difference. It is now recorded as unmeasured wherever it appears.
  **Fixed for future runs only:** the loop divides by the group's actual size. `renderer/training/tests/test_gradient_accumulation.py` asserts an even split equals the single-batch gradient, that the trailing-group correction differs from the legacy form by exactly half that group's gradient, and that `accum_steps: 1` is bit-identical to the unaccumulated path. **The committed M5 runs keep the behaviour they were trained under** — changing their numbers retroactively would be worse than the asymmetry.

- **Protocol for future ladder comparisons**, defined here and deliberately **not** applied to historical results: rungs must match on **samples seen per epoch** and **optimiser updates per epoch**, not merely on nominal effective batch. Where `drop_last` and batch size make those differ, either equalise them by construction or record the residual as a measured quantity. Applying this rule backwards would mean retraining, which this pass explicitly does not do.

- **Statistical language corrected in three places.** (i) Decision 46 claimed the three-seed spread showed "training does not reproduce exactly" — a **category error**: all three runs used *different* seeds, and same-seed reproducibility was never tested and remains untested. (ii) The spread is n = 3 and **pools a toolchain change** (seed 20260905 under `.venv`/torch 2.14.0+cu130; the others under conda `dl`/torch 2.13.0+cu132), so it is a **preliminary estimate of CNN run-to-run spread** — not a confidence interval, and **not a measure of StructFormer variance**, which has never been measured. (iii) The sharpness band no longer licenses "one run settles it": it was the operational rule for deciding whether to buy more GPU time, and the conclusion is scoped to *these implementations, recipes, scene and style*. The pre-registered thresholds are retained as **historical operational rules** — fixed before the runs, which is their value — not as statistical guarantees.

- **Evaluation and deployment limits made explicit in the artifacts, not only the prose.** Both M5 metrics reports now carry a `selection_set_note` (these are best-of-N-epochs on the selection split, **not** an independent test result — no untouched test split exists at M5) and a `comparison_scope` recording the optimiser-exposure difference. Latency is labelled attempt-1, batch 1, 512×288, **engine time only**, excluding transfers and the game pipeline; p95 is marked *not measured* rather than left as a null that reads like a measurement. FP16 parity remains flagged as exceeding the FP32 tolerance, and the tolerance is **not** relaxed.

- **Third CNN seed's evidence recovered and committed.** Seed 20260909 — the run that widens the sharpness spread and therefore sets the M5 margin — existed only as a row in a summary file. Its config, training report, metrics and log were located in the session scratchpad and are now committed as `configs/training/m4_unet_seed3.json` and `eval/reports/m4_train_seed3.json` / `m4_metrics_seed3.json`. The checkpoint is 30 MB and gitignored, so its path and SHA-256 prefix (`2a98099886860bb9`) are recorded instead; the training console log is `.log` and therefore also gitignored, with the same history available inside the committed train report. **Nothing was reconstructed or guessed.**

- **Remaining evidence gaps, stated rather than closed:** the M5 export report is not yet regenerated end-to-end; Model A's TensorRT failure is undiagnosed; StructFormer run-to-run variance is unmeasured; same-seed reproducibility is untested; p95 latency is unmeasured; and the M3 teacher-gate figures still have no committed artifact (Decision 44).

- **No historical verdict changes.** M5 still fails "better or competitive against the CNN baseline" on both attempts, and still meets the other three exit criteria. What changed is the confidence with which that failure may be stated: it is one implementation, one recipe, one scene, one style, with unmeasured optimiser-exposure differences and unmeasured StructFormer variance — not a general result about window attention.

---

### Decision 52: Five Review Findings — A Report Path That Overwrote Committed Evidence, and a Mask That Only Discouraged (2026-09-08)

Second cleanup pass, continuing Decision 51. No training, no third attempt, no threshold change; both M5 attempts and Decision 49's verdict are unchanged, and no committed report was rewritten. Two of the five are material behaviour changes and are recorded as such.

- **`report_path()` silently inherited M4's destinations, and 7 of 9 config/mode combinations collided with committed evidence.** Verified by resolving them: `--mode train` on either seed-variant config resolved to `eval/reports/m4_train.json`, and `--mode overfit` on the M5 warmup config to `m4_overfit.json`. Model A's committed train/metrics/overfit artifacts were one un-flagged CLI invocation from being overwritten — the seed runs survived only because `--report` was passed explicitly each time. **Only `config_version: "m4-unet-1"` may now inherit the unprefixed names**; every other config must declare `reports`, and `assert_reports_resolvable()` resolves every destination a mode will write **before** a model is built or data is touched, so an undeclared path or an unsupported `--mode` aborts at startup rather than after a 40-minute run. The two seed configs now declare their own paths. The warmup config deliberately declares no `overfit` destination, so that combination is now **rejected** rather than silently redirected.

- **Entity masking discouraged illegal keys; it did not exclude them. (Material behaviour change.)** The additive `MASK_NEG = -100` is a *penalty*: once legal logits fall below it, illegal keys dominate the softmax and one entity's memory reaches another entity's pixels — a direct violation of §4's `p ∉ S_i` rule and of M8's whole premise. Replaced with masking to `-inf` on both the manual and SDPA paths, so an illegal weight is exactly zero for any finite logits. Fully-masked rows are given a uniform row before softmax and gated to exactly `0.0` after projection, which keeps them finite without reintroducing the NaN that Decision 45 item 2 records. Tests drive the real module at logit scales 1/50/500 and perturb a **single token component** or scale it — adding a constant to every component is removed by the module's LayerNorm and would pass regardless of the masking. Verified by re-injecting the additive form: **6 tests fail**, precisely at large logits and padded slots.
  **The module stays probe-only and is not wired into M5** (asserted). Its ONNX graph changed (`Where`/`masked_fill` in place of an `Add`), so CPU ONNX parity is now covered; **TensorRT behaviour after this change is UNVERIFIED** and needs a GPU measurement. `eval/reports/m5_export_probe.json` describes the **old** additive form and is preserved unchanged as the historical record.

- **`execute_async_v3` returns `False` on failure; both loops ignored it.** A failed enqueue therefore produced a published latency and `run_ok: True` for an engine that never ran. `execute_or_raise()` now raises with the phase and iteration, in warmup and timed loops alike.

- **Engine input shapes were never checked.** Binding is by raw device pointer, so TensorRT cannot detect a mismatch: an undersized buffer reads out of bounds, and an equal-element-count reshape — `(1,7,512,288)` for `(1,7,288,512)` — is silently misread as valid data. `validate_input_shape()` runs before any pointer is bound.

- **The export probe's gate omitted required outcomes and always exited 0.** `blocked` counted only ONNX export and FP32 run, so an FP16 build or run failure, or an FP32 parity failure, left it empty and the probe reported success. `_blocked()` now covers FP32 build/run/parity and FP16 build/run, rejects non-finite parity values, and `main()` returns **1** when any variant is blocked while still printing full diagnostics. `--no-trt` is explicitly scoped in the artifact and on stdout: an empty `blocked` list from an ONNX-only run does **not** assert that TensorRT passed. No FP16 tolerance was introduced or relaxed.

- **Tests exercise the production helpers**, not copies of them: `_blocked` / `_finite` are imported from the probe script, `execute_or_raise` / `validate_input_shape` from the runner, and the masking tests instantiate the real module. 307 CPU tests pass; no GPU work was run and no historical artifact was regenerated.

---

### Decision 53: RCS Implemented and Measured — The Headline Metric Does Not Degrade With Time Unseen (2026-09-09)

§11's headline figure, **RCS versus time outside view**, had never been computed. Six milestones were complete and `eval/` contained only `image_quality.py`. Implemented in `eval/metrics/rcs.py`, run by `eval/protocols/run_rcs.py`, artifact `eval/reports/rcs.json`. Measured on the existing M4 and M5 checkpoints — **no training**.

- **Three choices §11 does not fix, made here.** `E` is a **masked deep-feature descriptor**: AlexNet conv taps (the LPIPS trunk, already a dependency), average-pooled over the entity's pixels, per-layer L2-normalised, concatenated. The mask always comes from the **engine's** `entity_id` map, never from the rendered image, so it cannot move with a render's own errors — `entity_id` is used as an address here, exactly as Decision 9 requires. There is **no fallback descriptor**: if the trunk is unavailable the caller gets an error, never a home-made number published under the name RCS (Decision 35's rule).

- **The first implementation was confounded and the first numbers were wrong.** Masked *pooling* alone does not exclude background: conv features have receptive fields, so pixels around the silhouette reach the descriptor. Measured — replacing the **entire** background scored **0.979** while inverting the **entity itself** scored **0.971**, comparable magnitudes, i.e. the descriptor was reading the scene nearly as much as the entity it was supposed to describe. Fixed by zeroing the background **before** the trunk, after which the same background swap scores exactly **1.000**. Caught by a test written to be non-vacuous rather than by inspection, and the pre-fix numbers are not reported anywhere.

- **⚠ The figures in this entry are the SINGLE-FRAME-PAIR measurement, superseded by Decision 54.** Each sequence was scored on exactly one pair — `last_seen_frame` against `first_seen_again_frame` — so one unlucky pose moved a whole data point out of only ten. Decision 54 re-measures over a 3-frame window each side (90 pairs per source). The ordering and the central finding are unchanged; the values below are not the current ones. Read `eval/reports/rcs.json`.

- **The measured baseline.** Ten reappearance sequences — 0.5/1/2/4/8 s × `camera_away`/`entity_away`, built by M2 for exactly this. The **native engine** scores **0.973** (the ceiling that pose change alone allows; not 1.0 because the object is seen from a different angle after the gap). The **teacher targets** score **0.918** — *less* consistent than the engine, so the supervision itself injects inconsistency. All three trained models land at **0.926–0.927**, indistinguishable from one another against a per-source sd of ~0.02, and slightly **above** their own teacher.

- **The finding that matters: RCS does not degrade with time outside view.** Across a 16× range of unseen durations the change from 0.5 s to 8 s is **−0.012 to −0.028** for the models and **+0.000** for the engine, against a sd of ~0.02 — inside noise, and non-monotone (2 s is the worst rung for most sources, including the engine, so it is a property of that rung's pose, not of elapsed time).
  **This is expected by construction and was not previously stated.** A frame-independent renderer computes appearance from the current frame's structure alone; it has no representation of how long an entity was away, so its RCS *cannot* be a function of elapsed time. It is a function of pose difference.

- **Consequence for M8, decided before EPAM is built.** §11 says "a strong EPAM result should degrade materially slower as time unseen increases". **That framing presumes a baseline that degrades, and against the frame-independent rungs no such baseline exists** — a slope comparison against Models A or C is unwinnable because there is no slope to beat. Two consequences:
  1. EPAM's demonstrable claim on this dataset is a **level** improvement — closing the gap from **0.926** toward the engine's **0.973**, about **0.047** of headroom. With model-to-model sd near 0.02, EPAM must move RCS by roughly 2 sd to be credible. That is a demanding bar and it is now known in advance rather than after the fact.
  2. A **slope** effect can only appear against a model that carries state across frames and can therefore drift — the **temporal** rung (M6). If the slope figure §11 describes is wanted, M6 is its baseline, not M4 or M5.

- **A confound to carry forward.** The models score *above* their teacher, and blur is the likely reason: averaging suppresses the high-frequency variation that differs between two views. The same softness that failed M5's sharpness gate plausibly *helps* RCS. So RCS must never be read alone — it is paired with the sharpness ratio, or a blurrier model looks like a more consistent one.

- **Scope.** One session, one style, one entity per sequence, n = 10, and a descriptor whose absolute scale has no meaning outside comparisons made with it. The ordering (engine > models > teacher) and the absence of a time trend are the results; the third decimal place is not.

---

### Decision 54: Strict Masking Reaches TensorRT; RCS Re-measured Over Frame Windows (2026-09-10)

Two verifications closing gaps Decisions 52 and 53 left open. No training, no new milestone.

- **The strict entity mask exports and runs under TensorRT.** Decision 52 replaced the additive `-100` penalty with masking to `-inf` and recorded the TensorRT behaviour as **unverified** — a real risk, because that module is M7's and M8's core mechanism. Verified now on both attention paths: ONNX export succeeds, shapes stay static, the engine builds and runs, parity is far inside the 1e-3 tolerance, and — the failure that actually mattered — **the outputs are finite**. The `-inf` mask, uniform-row substitution and output gate survive the export boundary without producing NaN. Evidence: `eval/reports/m7_mask_strictness_trt.json`, written by `scripts/verify_mask_strictness_trt.py`. The historical probe `m5_export_probe.json` describes the previous additive form and is **unmodified**.
  **Five operators are new** relative to M5's declared allow-list — `Greater`, `Not`, `Or`, `ReduceSum`, `IsNaN` — from computing `row_has_legal`, the uniform-row substitution, and SDPA's own NaN handling. All are TensorRT-supported and all ran. They are deliberately **not** added to `ALLOWED_OPS`, because this module is probe-only and is not in M5's shipped graph; they must be added when it actually ships at M7. Recorded so that is a visible diff and not a surprise (§12).

- **RCS now averages over a frame window, and the finding survives it.** Decision 53's measurement used a single pair per sequence — ten samples in total, where one unlucky pose moved a whole data point. `evaluate_source` now walks up to three *contiguous visible* frames outward from each boundary (stopping at the gap rather than jumping it) and averages every before×after pair: **90 pairs per source**, with a within-sequence spread recorded alongside each score.
  Every score falls slightly, which is expected — a wider window admits more pose variation — and the **ordering is unchanged**: engine > models > teacher, with the three models still indistinguishable from one another and still slightly above their own teacher.

- **The central finding is stronger, not weaker.** Under the single-pair measurement the 0.5 s → 8 s change was −0.012 to −0.028, which was inside noise. Under 90 pairs it is **zero or slightly positive for four of five sources**. There is no degradation with time outside view to be found, and the 2 s rung remains the worst for **every** source *including the native engine*, which is the control that settles it: that dip is the rung's own geometry, not elapsed time and not any model's behaviour.
  So Decision 53's consequence for M8 stands and is now better supported: against frame-independent rungs there is **no slope for EPAM to beat**, and EPAM's demonstrable claim on this dataset is a **level** improvement toward the engine's ceiling. The current headroom and the spread it must overcome are in `eval/reports/rcs.json`; deliberately not retyped here, per Decision 42.

---

### Decision 55: M6 Warp, Mask and Gate — The Validation Split Cannot Measure Reprojection (2026-09-10)

Milestone 6 steps 0-3: engine-guided warp, confidence mask, and the temporal-error baseline, all measured **before Model D is trained** so the exit criterion cannot become a hindsight judgement. Nothing was trained. Evidence: `eval/reports/m6_gridsample_probe.json`, `m6_mask_calibration.json`, `m6_temporal_baseline.json`, each written by a committed generator (Decision 51's rule).

- **`GridSample` reaches TensorRT — M6's biggest export risk, cleared first.** ONNX gained `GridSample` only at opset 16 and it is a known-risky operator, so it was probed **before** `renderer/temporal/` existed, the same sequencing that paid for itself at M5. Every variant (bilinear/nearest, zeros/border, `align_corners` both ways, image and feature resolution) exports at opset 17, **builds and executes** under TensorRT, and matches PyTorch far inside the 1e-3 tolerance. Marginal cost is **~0.017 ms** against M5's 4.22 ms FP16 engine — negligible against §12's 15-20 ms allocation.

- **M1's motion convention composes with M5's architecture unchanged, and this was verified rather than assumed.** Screen-UV in `[0,1]` is resolution-independent, so the field needs **no rescaling** at feature resolution; backward is exactly the direction a warp samples; and Y-down UV maps onto `grid_sample` with a plain `2*(uv + motion) - 1` and **no Y flip**. Decision 25's contract needed no adaptation.

- **The confidence mask is calibrated from measured distributions, not from invented constants.** §4's `M_t = exp(-alpha*e_D - beta*e_N)` is given; the constants are not. `e_D`'s half-point is **0.175 m**, the Youden-optimal separation between the scene's measured "same surface" and "different object" neighbour-step distributions — and the separation curve is **flat over 0.10-0.35 m**, so the third digit carries no weight and must not be re-tuned to move a result. `e_N`'s half-point is **1 - cos(30 deg)**: a 30-degree orientation change is a change of facet, and a 90-degree cube edge lands at ~0.006, effectively zero. **Five** cases are hard-zeroed because no exponential can express them: out-of-bounds samples, `motion_valid == false` (frame 0), background/sky where motion is pinned to `(0,0)` by convention, **M1 scenario S7** — an entity spawned this frame also carries `(0,0)`, which the exponential cannot distinguish from "static" — and a **warped source that was itself background**, because §4's `e_D = |D_t - W(D_{t-1})|` is undefined when *either* operand is the `+inf` sky sentinel. ⚠ An earlier version of this bullet said four and omitted the last; the code has always implemented five (`renderer/temporal/confidence.py`). Corrected per Decision 58.

- **The finding that reshapes the exit criterion: the validation split cannot measure reprojection.** 43 of its 44 `dt == 1` pairs displace by under **half a pixel** (median 0.26 px), because the held-out sequences are the two 8-second reappearance rungs and those park the camera — Decision 35 recorded that property for the qualitative strips and it lands here too. On that split temporal error is very nearly `|I_t - I_{t-1}|` under another name, and **warping makes it slightly worse** (gain ~0.95) because bilinear resampling costs more than a quarter-pixel of alignment buys. §10's criterion as written would therefore be won by whichever model **flickers least**, not by whichever reprojects best.
  **The warp is not the problem.** It reduces error **8.8x** on frames that actually move, and 2.04x in its unit tests. The measurement set is.

- **Chosen gate: guard-band pairs with at least 1 px of motion.** The `guard_frames = 60` band drops 180 frames — three contiguous ranges, 177 consecutive pairs — from **both** splits, so no model trained on them and no checkpoint was selected on them: they are genuinely held out, and unlike the validation rungs they carry the 4-second rungs' real motion. Filtering to >= 1 px leaves **28 pairs**. The threshold is chosen on physics and fixed before Model D exists: below ~1 px the bilinear resampling error exceeds what alignment buys, which is precisely why `warp_gain` is **below 1** on sub-pixel pairs even for the deterministic engine. Validation flicker is still reported alongside, for comparability with the rest of the ladder. **The locked split is untouched**, so M4/M5 image-quality comparability is unaffected.

- **Pre-registered threshold, fixed now.** Model D must reach a temporal error at least **2 CNN across-seed standard deviations** below Model C's on that gate, **and** must not lose more than 1 sd of sharpness against Model C on the same pairs. Values are in `m6_temporal_baseline.json -> m6_gate.pre_registered_threshold`; deliberately not retyped here (Decision 42). The spread is the same kind of quantity Decision 51 scoped: **a CNN measurement over 3 seeds that pools a toolchain change — a preliminary estimate, not a confidence interval, and not a measurement of StructFormer variance**, which remains unmeasured.

- **Two results worth carrying forward.** The **teacher targets are the least temporally stable source measured** — roughly 28x the engine's error on the validation split — because the diffusion teacher generates each frame independently; every trained model is *steadier than its own supervision*, which is what a frame-independent regressor's averaging buys and what M6 must improve on deliberately. And the blur confound Decision 53 found for RCS **did not materialise here**: Model C is both blurrier *and* worse on temporal error, so this metric is not merely tracking softness — but the sharpness guard is pre-registered anyway, because the confound is real in principle and Model C already sits below M4's 0.90 floor.

- **Recorded limits.** Two `dt > 1` validation pairs are dropped by the mask's minimum-kept-pixels rule and are **omitted, never imputed**; the counts are in the artifact. The gate's 28 pairs are a small sample, and the sharpness figures on that subset are per-pair means over moving frames, so they are comparable **within** the gate and not against the whole-split values quoted elsewhere.

---

### Decision 56: Model D Trains From Scratch — Decision 19's Staged Resume Is a Warm Restart Here (2026-09-10)

Deviation from **Decision 19**, which mandates resuming each stage from the previous stage's best checkpoint. Recorded rather than applied quietly, per CLAUDE.md.

- **Ladder integrity.** Model C trained from scratch for 60 epochs. Had Model D resumed from C and trained 60 more, D would have had **120 epochs of optimisation against C's 60**, and the M6 gate cannot separate that budget difference from the architecture it exists to measure.
- **And it does not work anyway.** Measured: initialised from C, the first batches sit at C's converged L1 (~0.005-0.007); after **two** optimiser steps at `lr = 2e-4` with no warmup, L1 is ~0.07 — an order of magnitude worse. AdamW's first bias-corrected step has magnitude ~`lr` regardless of gradient scale, so it perturbs a converged 8 M-parameter solution rather than refining it. It is a warm **restart**, not a fine-tune, and Decision 19's efficiency rationale does not apply.
- **Every way to keep it adds a second difference against Model C** — warmup, a lower fine-tune lr, or a longer schedule. Warmup specifically was tested at M5 attempt 2 and came out worse on every metric (Decision 49).
- **Result:** Model D trains from scratch on Model C's exact seed, lr, weight decay, epochs, effective batch, split and selection rule. The only differences are the temporal fusion and the temporal loss term.

---

### Decision 57: M6 Negative Result — Temporal Reprojection Does Not Separate From StructFormer Seed Noise (2026-09-10)

**Milestone 6 does not pass.** All three deliverables exist and work; the exit criterion — "lower temporal error than frame-independent StructFormer" — fails against the margin pre-registered in Decision 55 before Model D existed. The threshold was verified byte-identical before and after Model D was scored; **no goalpost moved**.

- **The first reading was too kind, and the corrected one is decisive.** Against Model C's committed run, Model D improves temporal error by **~8 %** — consistently, on every set measured (the gate, the full guard band, validation flicker, and the out-of-distribution `dt > 1` pairs). Against the required 15.4 % that read as a near-miss. (Measured per set: −7.9 % on the gate, −9.3 % on validation `dt == 1`, −9.8 % on the full guard band — a 7.9–9.8 % range, not a uniform 8 %.)
  Then StructFormer's **own** across-seed variance was measured for the first time — two further Model C seeds, everything else at parity. **One Model C seed beats Model D outright**, and Model D lies *inside* Model C's own seed range. The earlier figure was measured against a comparatively unlucky single draw. Evidence: `eval/reports/m6_structformer_variance.json`.
  **The honest statement is therefore not "improved but short of the margin". It is that engine-guided temporal reprojection produces no improvement separable from StructFormer's run-to-run noise on this dataset.**

- **The corrected margin was accepted in the direction that hurt, as pre-committed.** The original margin used **CNN** spread as a stand-in because StructFormer's had never been measured — a substitution Decisions 51 and 55 both flag. Measuring the right quantity was pre-committed to be binding either way, and the stated expectation was that a transformer would prove *less* stable and the bar would get **harder**. It did: StructFormer's relative across-seed sd is **9.07 %** against the CNN's 7.7 %, so the margin widens to 18.1 % and M6 fails by more. Reference is the three-seed mean, not a single draw (Decision 46's construction). **Against that mean Model D is −5.5 %, against a required −18.1 %** — the two figures a reader should pair; quoting −7.9 % beside −18.1 % mixes reference points.

- **Model D did not win by blurring, and that mattered.** The sharpness guard passes with room — Model D is *sharper* than Model C on the gate's pairs. That guard existed only because a softer renderer earns lower temporal error for free (Decision 53's RCS confound on a new metric), and it did its job by ruling the cheap explanation out rather than by catching anything.

- **What is verified and reusable regardless.** `GridSample` builds and runs under TensorRT at ~0.017 ms; the warp reduces error **8.8x** on frames that actually move; the confidence mask is calibrated from the scene's own measured distributions with hard zeros for out-of-bounds, frame 0, background and M1's S7 spawn case. M7 and M8 inherit all of it.

- **Stated against the result, not buried.** Model D carries **+5.9 %** parameters over Model C (the fusion convolution) — a real fraction of any margin it is judged against. Its per-frame image quality is slightly **worse** than Model C's (val PSNR 32.675 vs 32.807). And `attempt1` was trained with `num_workers = 6` while the two variance seeds used 2, reduced after a system OOM kill; worker count does not enter the computation, but it is a difference between runs pooled into one standard deviation and is recorded rather than left unsaid.

- **Three results now point the same way.** M5: StructFormer does not beat the CNN at parameter parity. Decision 53: RCS shows no degradation with time unseen for a frame-independent renderer, so there is no slope for memory to improve on. M6: temporal conditioning does not separate from seed noise. All three are consistent with **644 training frames** being the binding constraint rather than any architectural choice — against AlayaRenderer's 4 M+ frames and a 1.3 B fine-tuned backbone (Decision 45's context). **This is recorded as a pattern to weigh before M7/M8, not as a conclusion that any mechanism is wrong.** No third tuning attempt is proposed.

---

### Decision 58: M6 Review — Four Flagged Deviations the Log Never Absorbed, and a Pre-Registration That Contained Its Own Subject (2026-09-10)

From the M6 spec-guardian pass. The negative result of Decision 57 survived review on substance — the threshold is internally consistent, the pre-commitment was honoured in the direction that hurt, the decisive seed-overlap observation is true and prominent, and every locked decision tested (4, 8, 9, 21, 25, 41, 47, 52) holds under tests that *measure* rather than assert. What failed was the evidence discipline around the recording, which for a second consecutive negative result is the part that matters most.

- **The artifact carrying the verdict had no generator and no test.** `m6_structformer_variance.json` — the file on which Decision 57's entire verdict *change* rests — was referenced by the KB and by nothing else. This is precisely Decision 51's finding, recurring in the entry that cites Decision 51. `scripts/measure_structformer_variance.py` is now its only writer, with tests asserting provenance, three distinct seeds, and that the corrected margin is **wider** than the CNN stand-in it replaced (a narrower one would be threshold-shopping and must fail loudly).

- **The pre-registration artifact contained the model it pre-registers against.** `run_temporal_baseline.py` both computes the gate *and* scores every rung including Model D, writing to the same path — so the committed `m6_temporal_baseline.json` ended up holding `m6_temporal_model_d`, while its own `purpose` field still said the measurement was taken "before Model D is trained", and the M6 config asserted a guarantee about that file the generator did not honour. Decision 57's claim that the threshold was "byte-identical before and after" was **true but unverifiable from the repository** — the evidence was in a temp file. `eval/reports/m6_temporal_gate_prereg.json` is now the committed frozen copy, containing no Model D, and a test asserts both that absence and threshold equality with the live artifact.

- **Two committed artifacts disagreed on the headline number.** Model D's gate temporal error appeared as `0.0032416` in one and `0.0032424` in the other — a ~0.025 % difference from GPU non-determinism across two independent computations of one quantity. The verdict is unaffected (both fail both thresholds), but two computations of one headline number is how artifacts drift apart. The variance generator now **reads** Model D's figure from the gate artifact instead of recomputing it.

- **Four deviations were correctly flagged in code and never reached the Decision Log.** CLAUDE.md requires a dated entry, not a code comment. Recorded here:
  1. **§4's `L_temp` has no gradient as written.** It uses the un-hatted `I` on both sides, making the term a constant of two teacher frames. The prediction is substituted on both sides. The alternative — anchoring to the *warped teacher target* — is rejected on measurement, not taste: Decision 55 found the teacher the least temporally stable source measured (~28× the engine), so that objective would train the model toward instability.
  2. **A fifth hard zero exists** (warped source was background), because `e_D` is undefined when either operand is the `+inf` sky sentinel. Decision 55 said four; corrected above.
  3. **`e_N` is not renormalised after warping** — a deliberate literal reading of §4, previously undocumented.
  4. **S7's hard zero has no data-backed exercise on this session** — it is unit-tested only, which is a stated scope limit on M6's mask deliverable rather than a defect.

- **A live trap: `train_baseline --mode eval` would have scored Model D frame-independently and published it.** `TemporalStructFormer.forward` accepts `model(x)` because cold start legitimately needs it, so every mode of the frame-independent trainer would have run Model D **without history** — measuring a model that does not exist — and written it to `eval/reports/m6_temporal_metrics.json`, a destination the M6 config declares and `assert_reports_resolvable` therefore permits. Nothing had been mis-measured, because a stray constructor key in the architecture registry happened to raise first. That accident is not a guard: `train_baseline` now **refuses** `arch == "temporal_structformer"` with a message naming `train_temporal`, the stray key is removed, and a subprocess test asserts the refusal. `load_checkpoint` still handles the arch, because the M6 evaluator needs it.

- **Three stale rationales asserted what Decision 56 disproved.** The config and two docstrings still described `init_from` as "a true continuation rather than a perturbation" — the zero-initialised fusion conv was the *reason* that was believed, and Decision 56 showed the perturbation comes from AdamW's first bias-corrected step instead. Corrected in place. Likewise `temporal_model.py` claimed its forward signature *prevented* frame-independent use; it does not, and the guard now lives in `train_baseline` where it belongs.

- **Scope corrections to Decision 57's own figures**, made rather than defended: the improvement is **7.9–9.8 %** across sets, not a uniform ~8 %; the corrected comparison is **−5.5 % against a required −18.1 %**, and those are the two figures to pair (quoting −7.9 % beside −18.1 % mixes reference points); and the ~0.017 ms `GridSample` cost is the *minimal standalone* variant, not the shipped warp-and-mask graph.

- **Still open, stated rather than closed:** `w_temporal = 0.5` was never swept — deliberately, since tuning it against a pre-registered gate would convert the gate into a fitted criterion, but it means M6's negative result is scoped to one untuned weight. Model D has **no ONNX/TensorRT evidence**, and its runtime would need to carry a `(1, 480, 36, 64)` feature tensor across frames — state the M5 runtime never had, and an M10 consequence with no entry of its own yet.

---

### Decision 59: M6 Export Evidence — Model D Deploys, at 47 % More Latency and With Cross-Frame State (2026-09-10)

Model D shipped with its export block marked `NOT YET EXERCISED`. §10 does not list export among M6's deliverables, but §12 requires the path proven, M10 depends on it, and every other ladder rung has it — so the milestone was incomplete regardless of its verdict. Evidence: `eval/reports/m6_onnx_export.json`, written by `scripts/export_benchmark_m6.py` and asserted by `test_m6_reports.py` (Decisions 51/58's rule, applied from the start this time rather than retrofitted).

- **Both graphs export, build and run.** Opset 17, fully static shapes, TensorRT engines built **and executed**, outputs finite. The warp reaches TensorRT inside a real trained model, not just in the standalone probe of Decision 55.

- **Model D is a TWO-graph runtime, and that is new.** The `encoder` produces `prev_features (1, 480, 36, 64)` and must run **once per frame** to supply the *next* frame's history; the `renderer` is four-input — `(structure, prev_features, motion, confidence)`. **The runtime must therefore carry a 1.11 MiB feature tensor across frames, and the M5 runtime was completely stateless.** That is an M10 architectural consequence, recorded here rather than discovered during integration. Cold start passes no history and reduces Model D exactly to Model C.

- **The latency cost is real and is stated against a rung that failed its criterion.** Per frame in FP32: **19.80 ms** (encoder 6.20 + renderer 13.60) against Model C's single-graph **13.48 ms** — **+47 %**, and at the very **top** of §12's 15-20 ms StructFormer allocation. The cost is structural rather than incidental: a temporal rung pays two graph executions where a frame-independent one pays one. **No FP16 engine was built for Model D**, so the comparison against M5's 4.22 ms FP16 does not exist and must not be inferred — M5's FP16 required a separately exported fp16 graph (Decisions 45/48).

- **The encoder stage exceeds the FP32 parity tolerance, and is recorded as exceeding it.** 1.346e-03 against 1e-3. Context that is *not* a defence: the 1e-3 figure was written at M4 for a **sigmoid-bounded [0,1] RGB output**, and this stage emits **unbounded encoder features** (mean |x| 0.495, p99 1.623, max 3.111), so an absolute bound is the wrong instrument — the deviation is 8.3e-04 of the p99 magnitude. The `renderer` stage, which does emit [0,1] RGB, passes at 1.65e-04. **An intermediate-tensor stage needs its own relative tolerance before it can be gated; none is invented here**, and no M6 conclusion depends on this stage passing. A test asserts any exceedance carries its note, so a reader of the JSON alone cannot see a green stage.

- **The allow-list guard fired, and was obeyed rather than widened.** The first run was rejected for an unlisted `AveragePool` — the mask's area-pooling to feature resolution (Decision 55 chose area over bilinear because a feature cell covers 8x8 pixels and bilinear would report full confidence for a mostly-disoccluded cell). It was added deliberately, with that reason, which is exactly the "a new operator must be a visible diff" behaviour §12 asks for and Decision 51 rebuilt the list to provide.

- **What this does and does not change.** M6 still **fails** its exit criterion (Decision 57) — export evidence completes the milestone's record, it does not alter the verdict. What it adds to the ladder is a cost side to set against the quality side: Model D is not separable from Model C on temporal error, and is 47 % more expensive per frame with a stateful runtime. For M7/M8 that is the more useful half of the result.

---

### Decision 60: Cross-Frame Teacher Conditioning Is REFUTED — 11x Worse, Not Better (2026-09-12)

Decision 57 identified the teacher's temporal instability (~32-35x the engine's) as M6's root cause: a temporal rung trained on per-frame-independent supervision is taught to reproduce flicker, and its temporal loss is outvoted by reconstruction. The obvious remedy — condition each frame on the previous one — was **probed before being adopted, and it failed badly.** Evidence: `eval/reports/m3_temporal_teacher_probe.json`, written by `scripts/probe_m3_temporal_teacher.py`.

- **What was tested.** One variable. `init_t = M_t · W(styled_{t-1}, V_t) + (1 - M_t) · native_t` — seed frame *t*'s img2img from the previous **styled** frame warped forward by the engine's own motion, falling back to the native frame where M6's confidence mask says the warp is untrustworthy. Seed, prompt, strength, guidance and ControlNet scale unchanged. 30 frames of a moving span in the guard band.

- **The result, on the M6 gate's own metric:**

  | source | temporal error | vs engine |
  |---|---|---|
  | native engine | 0.000163 | — |
  | teacher, independent frames (current) | 0.005211 | 32x |
  | teacher, **cross-frame** | **0.061541** | **378x** |

  **+1081 %.** The remedy made the root cause an order of magnitude worse.

- **⚠ The explanation is a reading, not a measurement — and must not harden into one.** The number most consistent with 378x is **compounding re-stylisation**: at `strength 0.50` the pipeline re-denoises half the trajectory, so feeding it an already-styled frame applies the style a second time, then a third, down the chain; each frame departs further from the last and the frame-to-frame difference *grows* rather than shrinks. Supporting this reading: `native_engine` is unchanged at ~0.00016 across both runs, so the engine's motion is not implicated. **Not confirmed:** the per-frame images were not retained, so no drift curve was measured. If this entry is ever cited as "the reason we did not do cross-frame conditioning", the citation must carry this caveat with it. The *refutation* is measured; the *mechanism* is inferred.

- **What this rules out, and what it does not.** Ruled out: naive appearance propagation through img2img at this strength. **Not** ruled out: cross-frame conditioning in general — latent-space blending, a reduced strength on chained frames, and true cross-frame attention are all untested and would each need their own probe. No further variant is proposed here.

- **A second, more important finding about the current teacher.** While building the probe: the pipeline **already re-seeds `torch.Generator` with the same value on every frame**, so the initialisation noise is *identical* across frames and the flicker is not a noise artefact. It is inherent to per-frame img2img — small drifts in the conditioning push the denoising trajectory to a different mode. Any future attempt at teacher temporal stability must start from that fact; "fix the seeding" was already done and is not the lever.

- **What the probe bought.** ~20 minutes of GPU, against a regeneration of 990 targets, a re-run of Decision 32's four alignment gates, and a retrain of Models A, C and D — roughly a day — on a change that would have made supervision twelve times less stable. This is the third time in this project that probing before committing has changed the decision (Decisions 45, 55, 60).

- **Chosen path instead: more data, current teacher unchanged.** Three results now point at the 644-frame training set rather than at any mechanism (Decision 49's M5 result, Decision 53's RCS finding, Decision 57's M6 result). 2-3 further sessions are being recorded, targets regenerated with the **current, independent-frame** teacher config, and Models A, C and D retrained. **M5's and M6's failures are to be re-evaluated against the larger set before either is called resolved** — more data being *sufficient* is a hypothesis to test, not an assumption to act on.

---

### Decision 61: Fifteen Sessions — Viewpoint and Trajectory Diversity, Not New Scene Content (2026-09-13)

The 990-frame session every rung through M6 was trained on was recorded by a fully deterministic script: fixed camera sweep (-18..18 deg), fixed drift (`sin(u*TAU)*1.5`), fixed prop layout. Re-running it produces the same session, so "more data" required adding controlled variation first.

- **`STRUCTRENDER_SESSION_VARIANT`**, read by `seed_record.gd`. **Variant 0 reproduces the committed session exactly**, so `ds_1788687350_337657ea` stays regenerable; variants > 0 perturb the camera rig yaw, the sweep range, and the entity drift amplitude and phase from a variant-seeded RNG.
- **Prop layout is deliberately NOT jittered.** Decision 38's three placement constraints (cover both camera poses, never fully occlude the camera->target sightline across the drift envelope, sit 3-40 m) are hand-verified for the current arrangement, and `_check_one_hidden_run` runs in the CI fixture rather than the seed recorder. **So this buys viewpoint and trajectory diversity, not new scene content** — stated plainly rather than implied otherwise.
- **15 sessions, 14,850 frames, 15/15 VALIDATED with zero warnings**, 15/15 ALIGNED with zero failures. Decision 34's frame-diversity check is clean on every one, so the perturbation produces genuinely distinct content rather than near-duplicates.
- **Five sessions failed on the first attempt** at the recorder's 1800 s timeout, during the final 8-second rung. Not thermal (GPU 60 C, no throttling) and not progressive — the *last* variant completed normally afterwards. The cause was IO contention: 4 GB of session writes had filled the page cache. Re-recorded with a 3x timeout and a `sync` plus settle between runs; all five then passed.
- **The variant sessions score BETTER on style signal** than the original — `stylisation_hist` median 0.618 against the original set's 0.287 minimum. The camera-rig yaw puts more geometry in frame, which is what Decision 37 predicted would raise style strength. The concern that a variant might yaw toward emptier geometry and weaken gate 2 did not materialise.

---

### Decision 62: Early Stopping on Every Rung (2026-09-13)

- **Identical rule for all four rungs**, so each trains to its own convergence rather than to an arbitrary shared epoch count: patience **6 evaluations**, minimum relative improvement **0.1 %**.
- **Patience counts EVALUATIONS, not epochs**, so it means the same thing at any `eval_every_epochs`. `min_rel_delta` exists because `val_l1` wobbles at the 1e-5 level between evaluations; without it noise reads as improvement and the run never stops.
- **It cannot change which checkpoint is selected.** `best.pt` is still the lowest `val_l1` ever seen, so stopping early can only cost time. Verified by simulation before use.
- **It does introduce different training budgets between rungs** — A stopped at 41, B at 45, C at 41, D ran to 59 — and that is visible in every report (`epochs_run`, `stopped_early`, `best_epoch`) rather than hidden.

---

### Decision 63: Model B — the CNN Temporal Rung M6 Was Missing (2026-09-13)

M6 concluded "temporal reprojection does not help" from Model D versus Model C alone. That cannot separate *"temporal reprojection does not help here"* from *"temporal reprojection does not help a transformer here"*, because §10 Milestone 9's **Model B** — the CNN temporal rung — had never been built.

- **The temporal mechanism is character-for-character Model D's**: the same mask-gated residual, the same zero-initialised 1x1 fusion convolution, the same warp, the same confidence mask, the same area pooling of the mask to feature resolution. If it differed, A->B and C->D would not be the same delta and the comparison would be worthless.
- **Fusion point is the U-Net's bottleneck**, its encoder/decoder junction — the structural analogue of where Model D fuses.
- **Capacity deltas match**: A->B is +6.7 % (8,378,563 vs 7,853,763), C->D is +5.9 %. Both are only the fusion convolution.
- **Untrained Model B is bit-identical to Model A**, cold start and with history, exactly as untrained D equals C. Asserted.

---

### Decision 64: Session-Level Split — the Project's First Independent Test Set (2026-09-13)

Every number in this project before now was a **best-of-N-epochs minimum on the frames used to pick the checkpoint**. With one session that was unfixable: Decision 35 held out contiguous *sequences* within it and needed a 60-frame guard band to keep train and val non-adjacent. Fifteen independent recordings change the instrument.

- **11 train / 2 val / 2 test sessions** — 10,890 / 1,980 / 1,980 frames. Train is **17x** the 644 frames every earlier rung used. Frozen in `configs/data/session_split_v1.json`.
- **Test sessions were chosen by POSITION in the sorted session-id list, before any metric was computed**, so the held-out set cannot have been picked to flatter a result. `build_splits` asserts the three sets are disjoint and refuses any session lacking a VALIDATED or ALIGNED marker.
- **The guard band is gone and is not needed**: sessions are independent recordings, not adjacent frames of one continuous drift. Train, val and test share no frames, no camera path and no entity trajectory.
- **Deviation from Decision 35, recorded as one.** Earlier ladder numbers are **historical**: they were measured on a different split and are not comparable to these. Every threshold is re-derived from the new data rather than carried over.
- **Triples never span a session boundary** — Model D's `(t-2, t-1, t)` are built per session and only then concatenated, because frame `t-1` of one recording has no motion relationship to frame `t` of another.

---

### Decision 65: Adding Sessions Silently Repointed Every Historical Test (2026-09-14)

Recording the 15-session set broke two calibrated tests, and the cause was not the models or the data — it was `discover_session()`, which returns the **newest** `ds_*` directory by mtime. Every config and fixture that relied on it was suddenly resolving to a session recorded yesterday instead of the one its numbers were calibrated on.

- **What actually happened.** `test_output_is_not_blurrier_than_the_target` began scoring the **M4 checkpoint against a session M4 was never trained on**, and reported 0.896 against its 0.90 floor — a real number about the wrong pairing. `test_the_median_error_reduction_is_substantial` measured the warp on a different camera path and got 1.23x against a 1.5x threshold calibrated on the original. **Neither was a regression; both were the data moving under the test.**

- **Fixed by pinning, not by relaxing.** Eight historical configs now name `data.session_dir` explicitly, and the temporal test fixture pins the same session with a fallback. The thresholds were not touched — moving a threshold to fit data the test was never meant to measure is how a gate stops being a gate.

- **The general hazard, worth stating beyond this instance:** *any* newest-wins discovery silently changes what a test measures the moment new data arrives, and it fails in the most misleading way possible — a plausible number about the wrong thing. Ladder v2 avoids it by construction: `data.split_file` names its sessions rather than discovering them.

---

### Decision 66: Model B's Temporal Fusion Is Inert; Model D's Is Not (2026-09-14)

`eval_ladder_v2.py` scores every rung through a frame-wise dataset that supplies no previous frame, so both temporal rungs ran **cold-start** — reduced to their own baselines. Their numbers there are not the temporal models'. `eval_temporal_ladder_v2.py` supplies the history and measures both paths **through identical code**, so the only difference between the two rows is the history.

| rung | cold-start L1 | with history | change |
|---|---|---|---|
| **B** (CNN temporal) | 0.008935 | 0.008935 | **+0.0 %** |
| **D** (StructFormer temporal) | 0.009490 | 0.009237 | **-2.7 %** |

- **Model B's fusion never engaged.** Identical to six decimal places with and without history: the zero-initialised fusion convolution never moved off zero. Model B is Model A plus 524,800 parameters that do nothing, and B's apparent edge over A in the image-quality table is not a temporal effect.
- **Model D's fusion works** — history improves L1 by 2.7 %, at a small sharpness cost.
- **A measurement error caught before it became a claim.** An earlier version of this comparison read cold-start sharpness from `style_acquisition` and history sharpness from a new local Sobel, and reported B gaining **+0.0465**. That was two different implementations, not an effect. Both rows now come from one code path. Recorded because a +0.0465 "temporal improves sharpness" would have entered the log as fact.

---

### Decision 67: M6 PASSES on Ladder v2 — the Margin Was Stale, Not the Result (2026-09-16)

**Milestone 6's exit criterion is met.** Evidence: `eval/reports/ladder_v2_temporal_error.json`, computed by `scripts/eval_temporal_error_v2.py` in a single pass that derives the threshold **before** writing the artifact, so the verdict is not a figure worked out by hand after seeing the numbers.

| | temporal error (330 moving pairs, sealed test sessions) |
|---|---|
| native engine | 0.000190 |
| teacher targets | 0.006137 |
| Model C seeds | 0.004155 / 0.004243 / 0.004539 |
| **Model D** | **0.003806** |

- **Margin: 2 x StructFormer's across-seed sd = 9.33 %. Model D is 11.74 % below the Model C mean. PASS.**
- **The rule did not change; the number feeding it was re-measured.** The margin had been carried over from the 644-frame dataset, where StructFormer's relative across-seed sd was **9.07 %** and the bar therefore ~18 %. Decision 64 required every threshold to be re-derived for ladder v2 and this one had not been. On 17x the data the spread falls to **4.66 %** — more data made StructFormer materially more reproducible — so the bar is 9.33 %.
- **Both sides moved, and that matters for how this is read.** D's improvement over C went from **-5.5 %** on the old data to **-11.74 %** here, and **D now beats every individual C seed**, including the best. On the old data one C seed beat D outright, which is what sank the earlier verdict (Decision 57). This is not the same measurement returning a different answer because the threshold moved.
- **Honest limits.** n = 3, so the sd is a preliminary estimate and not a confidence interval — the same caveat Decision 51 records for the CNN's. D clears the bar by 2.4 percentage points, which is real but not large.
- **The ceiling is unchanged and still governs.** The teacher targets remain **32x** the engine's temporal error, untouched by 17x more data, and Decision 60's probe refuted the obvious fix (cross-frame conditioning made it 11x worse). Model D at 0.003806 is still **20x** the engine. M6 passing means the temporal mechanism works *within* that ceiling — it does not lift it, and it does not weaken Decision 57's finding about the supervision.
- **M5 is unaffected and still fails.** On ladder v2's sealed test set Model C is +1.6 % worse than Model A on L1 and misses the sharpness gate at **0.8943** against the 0.90 floor. M6's criterion is temporal error; M5's is image quality against the CNN baseline. They are different questions and only one of them has flipped.

---

### Decision 68: The Recorder Does Not Wait on the Compositor (2026-09-20; written 2026-09-28)

> **Written retroactively.** This entry was cited by `seed_record.gd`, `d1_recorder.gd` and commit `1142267` from 2026-09-20 but never entered in this log; the number existed only in comments until 2026-09-28. The content below is taken from that commit and those comments, not reconstructed from memory.

**Problem.** Godot's vsync defaults to ENABLED, so every recorded frame blocked on a Wayland frame callback. With the screen on, that is ~60 fps and recording ran at **~9.6 files/s**. When the screen blanked or the window was occluded, the compositor throttled those callbacks and recording collapsed to **~0.27 files/s** — a **35x** slowdown, with the GPU parked in P8 and Godot at ~1 % CPU, which reads as a hang rather than a stall. The recorder never presents to a display, so it had no reason to wait.

**Decision.** Both recorders (`seed_record.gd`, and `d1_recorder.gd` for CI) call `DisplayServer.window_set_vsync_mode(VSYNC_DISABLED)` and set `Engine.max_fps = 0`.

**Why this cannot change the data — verified, not asserted.** The simulation is stepped by an explicit tick counter, never by wall-clock time or delta. The fix is applied to the CI recorder specifically so that its determinism suite exercises it: **286 checks pass, including 20 bit-identical frame comparisons across two runs** (commit `1142267`).

---

### Decision 69: Ladder v3 — 29 Sessions, and a Projection Pre-Registered Before the Result (2026-09-20; written 2026-09-28)

> **Written retroactively, and this matters for Decision 71.** Decision 71 states that "Decision 69 recorded a projection of 0.8979 before the run". This entry did not exist when that was written, so the KB cited a pre-registration that was not in the KB. The pre-registration itself is real and is verifiable from git: it is in the message of commit **`11422679f041d6643c5057f145285cf9e6bd1219`**, authored and committed **2026-09-20 23:58:15 +0530** and on `origin/main`. Model C (ladder v3) finished training at 00:18 on 2026-09-21, and the test-set sharpness that decided M5 was first reported in commit `ed661f1` at **2026-09-21 09:11:52 +0530**. The projection was therefore committed while Model C was still training, before any sharpness was measured. The quotation below is verbatim from that commit.

**Data.** 30 sessions were recorded (15 new). **One was dropped:** `ds_1789549998_8973903e` failed M2 validation with non-finite motion at frame 616 — the validator rejecting a bad session before it reached training. The 29 usable sessions split **21 train / 4 val / 4 test = 20,790 training frames** (`configs/data/session_split_v2.json`) — 32x the original 644 and 1.9x ladder v2. v1's held-out sessions stay held out, so the scaling points 644 -> 10,890 -> 20,790 are measured on the same frames at every size.

**Model A was retrained on the same data, deliberately:** best val_l1 0.009393 (v2) -> 0.009045 (v3), -3.7 %. The CNN improves with data too, so comparing a fresh Model C against the old Model A would have overstated the transformer by that margin.

**Pre-registered projection, verbatim from commit `1142267`:**

> Measured scaling: 0.8327 at 644 frames, 0.8858 at 10,890 -> +0.0188 per e-fold. At 20,790 train frames that projects 0.8979, which is 0.0021 SHORT of the 0.90 floor. The sizing exercise called for 23,198 training frames; allocating 4 val + 4 test sessions instead of 2+2 left 20,790. Better evaluation, less training data, and the target was not re-checked against it. Measured StructFormer sharpness sd across 3 seeds is 0.0079 — four times the projected gap — so the projection cannot distinguish pass from fail. Recorded here BEFORE the measurement so the prediction is on the record either way.

**Outcome (Decision 71):** measured 0.8514, further below the floor than the projection. The prediction was wrong in the opposite direction, and it is on the record either way.

---

### Decision 70: Product Scope — a Style Renderer, Not a Game Generator (2026-09-21)

Recorded because the intended deliverable was described in conversation but never written down, and the gap between what it says and what a reader assumes is large enough to mislead.

**What a user prompt controls: how the game LOOKS. Not what the game IS.**

StructRender re-renders an **existing Godot project** in a learned style. It does not generate geometry, levels, physics, mechanics, entities, or any game content. A user cannot type "make me a racing game" and receive one. This is not a limitation to be engineered away — it follows directly from the locked rule that **Godot owns all physics/logic/entity identity**, which is the same rule that lets EPAM key appearance memory to an entity that the engine *guarantees* is the same entity. Remove the engine's authority and the headline novelty has nothing to key on.

**The target interaction, scoped:**

> Play the game -> switch style -> the same world, same physics, same objects, repainted at ~4 ms/frame -> and the same barrel still looks like the same barrel across the switch.

The final clause is the contribution. It is what M8's *"style change clears memory"* exit criterion exists to verify.

**Three versions, and which one is being built:**

| | scope | status |
|---|---|---|
| **A — live switching among trained styles** | 2-4 styles, instant switching, EPAM clearing correctly | **this is what M8 builds** |
| B — new style from a new prompt | any prompt; ~4 h target generation + ~10 h training per style, offline | possible, documented as the extension path |
| C — unseen prompt, zero-shot | true generalisation to prompts never trained on | **out of scope** |

**Why C is out of scope, with numbers.** The renderer is a feed-forward network trained to *imitate* a diffusion teacher, not a diffusion model. It learned `oil_impasto` by copying 20,790 target images; it has no representation of what any other prompt means. Style tokens from a text encoder would let a new prompt produce a new token, but the model only learns what a token *means* from examples — with 1-2 styles it produces garbage for anything unseen. Zero-shot style generalisation realistically needs **20-50 trained styles**, which at ~4 h of target generation each is **80-200 hours before training starts**. That is not available on this hardware, and claiming it without the styles would be claiming a capability that was never measured.

**Consequence for the write-up.** StructRender must be positioned as a **renderer**, closer to a graphics setting than to a world model. The systems that do generate worlds from prompts need 4M+ frames and H200-class hardware (Decision 5's analysis) and hallucinate geometry — the exact failure engine grounding eliminates. That is a strength, but stated wrongly it reads as a shortfall against a capability never claimed.

**Open scope item for M8:** how many styles ship. Two is the minimum that makes the style token a genuine runtime input rather than a folded constant (Decision 45); more is better for the demo, and each costs ~4 h of generation.

---

### Decision 71: M5 Fails at 32x the Data — the Scaling Hypothesis Is Refuted (2026-09-21)

Ladder v3: 20,790 training frames, both rungs retrained on identical data, scored on 4 sealed test sessions (3,960 frames). Evidence: `eval/reports/ladder_v3_test.json`.

| | L1 | PSNR | sharpness | gate |
|---|---|---|---|---|
| **A** (CNN) | **0.008947** | **34.041** | **0.9104** | **PASS** |
| **C** (StructFormer) | 0.009174 | 33.858 | **0.8514** | **FAIL** |

**The prediction was wrong, and in the opposite direction.** Decision 69 recorded a projection of **0.8979** before the run. The measured value is **0.8514** — further below the floor than ladder v2's 0.8858.

| train frames | StructFormer sharpness | slope per e-fold |
|---|---|---|
| 644 | 0.8327 | |
| 10,890 | 0.8858 | **+0.0188** |
| **20,790** | **0.8514** | **-0.0532** |

**Sharpness decreased with more data.** The curve did not flatten, it reversed. The log-linear extrapolation that justified the scale-up was fitted to **two points**, which cannot distinguish a trend from a coincidence; that limitation was stated at the time and the projection was used anyway. ~13 h of compute tested the hypothesis, which is what testing is for — but the confidence attached beforehand was not earned by the evidence available.

- **M5's failure is not a data-volume problem.** Three sizes spanning 32x, all measured on the same held-out sessions, and StructFormer never approaches the floor while the CNN clears it at every size.
- **The CNN absorbs data better than the transformer.** Model A improved to 0.008947 / 0.9104 and now passes cleanly. The gap widened from **+1.6%** (v2) to **+2.53%** (v3).
- **Parameter parity held**: 7,853,763 vs 7,775,467 = **0.990x**, inside Decision 47's band. The comparison is not confounded by capacity.
- **Why sharpness fell is NOT established.** Candidates: the 4-session val set moved the early-stopping point (C stopped at epoch 31 vs 41 on v2), or L1 and sharpness trade off as L1 improves. This is **one run**; the measured seed spread is +/-0.0079 and the drop is 0.034, so noise alone is an unlikely explanation, but a single draw cannot separate the causes. Stated as unresolved rather than attributed.
- **Standing conclusion:** at parameter parity on a small custom dataset, a windowed transformer underperforms a U-Net. This is a finding about inductive bias, and it is better evidence than a rescued pass would have been.
- **What this does NOT touch:** M6 passed on ladder v2 (Decision 67) on a *different criterion* — temporal error, within the StructFormer branch. M5 and M6 ask different questions and only one of them fails.
- **The GAN question is now properly motivated.** Decision 4 marks adversarial loss optional-and-risky. Three scaling points showing data alone does not lift sharpness is a far stronger justification for trying it than the untested assumption that existed before.

---

### Decision 72: RCS Cannot Measure EPAM — Its Claim Moves to Sharpness (2026-09-21)

RCS computed on the trained ladder for the first time: 40 reappearance sequences across the 4 sealed test sessions, every source through one identical path, masks always from the **engine's** entity map. Evidence: `eval/reports/rcs_ladder.json`, `scripts/eval_rcs_ladder.py`.

| source | RCS | sd | vs native |
|---|---|---|---|
| **native engine** | **0.9283** | 0.0226 | ceiling |
| teacher targets | 0.8959 | 0.0383 | **-0.0323** |
| c_v2 (best model) | 0.9176 | 0.0250 | -0.0106 |
| a_v3 | 0.9120 | 0.0311 | -0.0163 |

**Two good findings.** Every model beats its own teacher (0.912-0.918 vs 0.896) — the renderers are *more* identity-consistent than the supervision they learned from, the same pattern as temporal error, and engine grounding is doing real work. StructFormer scores highest on RCS while losing on image quality.

**The finding that changes M8.** All six rungs are **statistically indistinguishable**: they span 0.0057 against 2 SE = 0.0093. Worse, headroom from the best model to the native ceiling is **0.0106**, so an EPAM effect would need to capture **87% of all remaining headroom** merely to become detectable.

**And nothing decays with time unseen** — the premise EPAM was scoped against:

| source | 0.5s | 2s | 8s | 8s - 0.5s |
|---|---|---|---|---|
| native | 0.9439 | 0.9086 | 0.9465 | **+0.0027** |
| a_v3 | 0.9287 | 0.8865 | 0.9319 | **+0.0031** |
| c_v2 | 0.9313 | 0.8954 | 0.9410 | **+0.0098** |

Every source rises slightly from 0.5s to 8s, and all share one shape (dip at 2s, recover at 8s) that therefore comes from *which scenarios those are*, not elapsed time. The gap to native is flat, not growing. **This is structural, not a scenario-design problem:** a frame-independent deterministic renderer cannot drift — same structural input, same output, regardless of how long the entity was away. Harder scenarios would not change it. Confirms Decision 53 at 4x the sample size.

**Where EPAM's value actually is — probed, not assumed.** `scripts/probe_averaging_cost.py`, no training required: take an entity across 3-4 consecutive frames at near-identical pose and compare individual teacher targets against their mean, on the entity's own pixels.

| | grad energy |
|---|---|
| individual teacher frames | 0.18396 |
| mean of those frames | 0.17259 |
| **cost of averaging** | **-6.2%, at 16.4 sigma** |

The teacher flickers 32x the engine; the renderer trains under L1+LPIPS whose optimum is the **conditional mean** of that flicker. The model is learning the blurry average by construction. That is a concrete mechanism for M5's sharpness failure, and "commit to one appearance per entity" attacks it directly.

- **Stated explicitly, because the distinction will otherwise be lost:** this measures *teacher frames averaged*, NOT EPAM's gain. It establishes the mechanism exists and is significant. It does **not** predict how much of the 6.2% EPAM recovers — the model averages over a far wider distribution than 3-4 frames, so the true cost may be larger, or EPAM may capture only part of it. "Averaging costs 6.2%" must not become "EPAM will gain 6.2%".
- **M8's pre-registered metrics, fixed before the model exists:** **sharpness ratio** is primary (the mechanism lives there); **RCS** is secondary and reported with the explicit note that it cannot distinguish an averaging model from a committing one; **cross-style RCS** is new — identity across a style switch, which the present single-style data cannot answer.
- **EPAM's scope is unchanged** (Decision: protected headline novelty). What changes is which measurement is expected to show it, decided from evidence before the milestone rather than after.

---

### Decision 73: The Residual Memory Probe — Information Exists, but Not at EPAM's Timescales (2026-09-21)

Run before M8 to test the assumption Decision 72 left standing: EPAM sharpens only if the part of the teacher that **structure does not explain** is correlated across time for the same entity. If it is independent, conditioning on memory narrows nothing, the L1+LPIPS optimum is the same blurry mean, and EPAM cannot sharpen. Evidence: `scripts/probe_residual_memory.py`, `eval/reports/probe_residual_memory.json`. Residual = `teacher_t - model_a_v3(structure_t)`, on the entity's own pixels. No training.

| lag (ticks) | residual correlation | sigma |
|---|---|---|
| 1 | **0.7172** | 47.4 |
| 2 | 0.5297 | 24.4 |
| 5 | 0.2870 | 12.2 |
| 10 | 0.1448 | 5.6 |
| 30 | **0.0366** | 2.0 |
| control: different entities, same frame | **-0.0080** | — |

The control sitting at zero validates the method.

- **Information exists, with a half-life of roughly 2 ticks.** By 30 ticks it is 0.037.
- **EPAM's reappearance gaps are 30-480 ticks** (0.5 s to 8 s at 60 tps). Across that entire range the residual correlation is at or below 0.037 — effectively nothing to retrieve. **A memory would return stale information.**
- **The exploitable window (1-5 ticks) is what temporal warping already covers.** Model D's one-frame history produced -11.74 % temporal error (Decision 67). EPAM would be reaching for signal that mechanism has already taken.
- **This does not contradict Decision 72's averaging result.** Averaging genuinely costs 6.2 % sharpness at 16.4 sigma — that mechanism is real. What this probe shows is that *memory across a reappearance gap* is not the lever that recovers it, because by then the residual is uncorrelated. Both findings stand; together they say the sharpness loss is real but memory cannot address it at the timescale EPAM operates on.

**A retraction, recorded so it does not resurface.** While interpreting this probe I claimed the 4 s and 8 s reappearance sequences were mislabeled, on the grounds that their frame gaps (30 and 60) implied 7.5 fps rather than 60. **That was wrong.** Long gaps are deliberately recorded at 1-in-8 (`REDUCED_STRIDE = 8`, documented in `dataset_harness.gd` and `dataset_manifest.md`), and 60/8 = 7.5 is exactly that stride. The per-frame `tick` field confirms the true gaps are **240 and 480 ticks = 4.00 s and 8.00 s**. Decision 72's analysis by `unseen_seconds` was correct; the "correction" — recomputing by frame index — was the error, since it lumped the 4 s sequence (30 frames, 240 ticks) with the 0.5 s one (30 frames, 30 ticks). **No decay with unseen time, to a genuine 8 seconds, stands as originally recorded.**

**Consequence for M8, to be decided before building:** EPAM's specified mechanism — persist appearance across an occlusion and restore it on reappearance — has no measurable information to carry in this data. The options are to re-scope EPAM toward where signal demonstrably exists (short-lag commitment, cross-style consistency), or to build it as specified and report a null result honestly. This is recorded now, before the milestone, rather than discovered inside it.

---

### Decision 74: M7 Complete — Structured Entity Awareness (2026-09-21)

All three exit criteria met, each asserted by a test rather than argued for. Code: `renderer/entity/` (keys, selection, pooling, model); tests: `renderer/entity/tests/test_entity_m7.py` (24) — suite now 477.

**Criterion 1 — correct entity mask/token association.** Pooling uses the entity's own pixels and nothing else, so a token describes its entity before masking is involved; disturbing one entity's pixels provably cannot move another's token. Masks land in **selection order**, unused slots are all-zero, and empty slots pool to exactly zero rather than NaN (dividing by a zero pixel count is the natural spelling and poisons every downstream token through the linears).

**Criterion 2 — no raw-ID production input.** Masks are binary indicators: `build_entity_masks` for `instance_id=3` and `instance_id=9999` produce **bit-identical** tensors, so the ID's *value* is gone by construction (Decision 9). Malformed keys raise rather than silently mis-address.

**Criterion 3 — deterministic overflow.** Descending `pixel_count`, ties broken by `(instance_id, generation)`. The tie-break is the part that makes "deterministic" real: equal pixel counts sorted by count alone would fall back to input order, which is exactly the arbitrary order the criterion forbids. Tested by feeding the same entities in different orders and requiring the same answer. An entity with no engine key is **dropped, not given a slot** — an entity the engine did not describe has no address, and inventing one would put an unidentified object in a memory slot.

**Model E — capacity parity.** `embed_dim=216` gives **7,428,507** params = **0.9554x** Model C (7,775,451), inside Decision 47's 0.9-1.1 band. Reusing Model C's `embed_dim=240` gives **1.1785x**, outside it. Width must be a multiple of 24 — GroupNorm needs 8, stage-1 attention needs `num_heads=3`.

  **Corrected 2026-09-21, same day.** This entry first claimed `embed_dim=224` gave **1.0272x**. That figure was computed with `num_heads=(4,8)`; Model C's config uses **(3,6)**, under which 224 is not even constructible (`dim 224 not divisible by num_heads 3`). The error came from sweeping widths against assumed hyperparameters instead of the ones in `configs/ladder_v3/model_c.json`, and it survived into a committed Decision because the sweep produced a plausible number and was not re-derived from the config. The corrected figure is above; the *lesson* Decision 47 encodes — that a two-sided band catches an overshoot a one-sided rule would miss — still holds, since 240 really is 1.1785x.

**A defect found and fixed during wiring, recorded because it is subtle.** Zero-initialising the *token* projection is **not** sufficient to start the entity branch at zero: `to_k`/`to_v` carry biases, so zero tokens still yield non-zero keys and values, and the attention's `proj` adds a bias of its own. Model E therefore differed from Model C at initialisation, which would have made "did entity awareness help?" inseparable from "was it initialised differently?". Zeroing the branch's output projection fixes it, and both directions are tested: E is identical to C at init **and** provably diverges once the branch is unzeroed.

  **Corrected 2026-09-22 — the Decision 66 citation here was inverted.** This entry originally cited Decision 66 as the reason zero-init is the *remedy*. It is the opposite: Decision 66 found Model B's **zero-initialised** fusion had never moved off zero, so zero-init is the mechanism that *failed* there. What caught it was measuring engagement **after training** — which Model E had not had at M7, since the rung was untrained. The engagement test at M7 hand-sets the weights and asserts nothing about training, despite its name. The Model B risk at rung E was therefore **deferred, not addressed**, and was only actually resolved when Model E trained (Decision 75: the branch engages, and the effect is a null).

**The TensorRT claim in this entry was overstated, corrected 2026-09-22.** `renderer/entity/model.py` said the mechanism was "TensorRT-verified at M5, precisely so that this wiring step could not discover an export failure after the fact". The artifact (`eval/reports/m7_mask_strictness_trt.json`) is **FP32 only** and probes `dim=32` at `18x32` — not Model E's bottleneck (448 wide, 36x64, `k_max=8`). It also carries an explicit instruction that five new ops "MUST be added deliberately when entity-masked cross-attention actually ships at M7", which was not done. **Model E has no export test.** Decision 75 has since replaced the attention with a lookup, which is strictly easier to export, but the gap is real and belongs to M9's export work.

**Where the isolation guarantee lives, stated precisely.** KB §4's `-inf` rule is asserted at the **attention output**, not at the final image. The decoder mixes spatially by design — convolutions and shifted windows deliberately spread information — so deleting a distant entity *does* change a given pixel's final colour, downstream of attention. A test asserting isolation on the output image would fail for a *correct* model, and "fixing" that would mean crippling the decoder. Measured directly: removing entity 1 changes entity 0's attention output by **0.000e+00**, while entity 1's own cells change by 1.33. §4 promises a token cannot reach a pixel outside its mask; that is what is tested.

**The probe-only guard was rescoped, not deleted.** It asserted no file may import `masked_entity_attention`. At M7 that is wrong — Model E imports it deliberately. What Decisions 45/50 actually protect is narrower: the **StructFormer files comprising the M5 graph** stay free of it so Model C remains the clean frame-independent baseline. A companion test now asserts Model E *does* import it, so a mechanism that is never wired in — the Model B failure — cannot pass silently.

**Not done here, deliberately:** Model E is built and tested but **not trained**. M7's exit criteria are correctness properties, not quality comparisons, and training rung E belongs with the ladder at M9.

---

### Decision 75: Entity-Masked Attention Degenerates to a Lookup; Model E Is a Null Result (2026-09-21)

**The attention was never attention on this data.** §4 masks attention to `-inf` outside an entity's own pixels, and §4 also defines `S_i(x,y) = 1[ID(x,y) = i]` — which is **disjoint**: a pixel belongs to exactly one entity. Measured (`scripts/probe_entity_attention_degeneracy.py`, `eval/reports/entity_attention_degeneracy.json`): **1,171 bottleneck cells hold a legal entity and 0 hold more than one.** A masked softmax over a set of size one returns **1.0 whatever the scores are**, so

    d(loss)/d(Q) = d(loss)/d(K) = 0, exactly and permanently.

`norm_q`, `to_q`, `to_k` — **374,976 parameters, 5.0 % of Model E** — received zero gradient at every step, still dead at step 11 while every other entity parameter had come alive. What remained was not attention: pixel *p* receives entity *i*'s value vector where *i* is *p*'s entity. That is a **lookup**.

`MaskedEntityInjection` is that lookup written as what it is. It is **bit-identical** to the attention path on disjoint masks (max difference `0.00e+00`, asserted by `test_injection_is_identical_to_masked_attention_on_disjoint_masks`) with the dead half removed — a simplification, **not** a behaviour change and **not** a departure from §4, whose `-inf` guarantee still holds exactly. The alternative — letting every pixel see every token so Q/K have something to choose between — would make attention meaningful by *violating* §4's masking rule, and was rejected.

- **Capacity parity, honestly.** Model E is now **7,053,547** params = **0.9072x** Model C (7,775,451), inside Decision 47's band, with **every parameter trainable**. The attention form's 0.9554x counted 375k parameters that could never learn.
- **A second undisclosed deviation, fixed.** Bottleneck masks were resampled with `interpolate(area)` then `> 0`, which **dilates**: any cell touched by one pixel of entity *i* became a full cell for *i*, so masks disjoint at image resolution **overlapped** at the bottleneck — pooling mixed entities and tokens reached pixels the engine gave to another. Measured on real data: **6 ambiguous cells**. Replaced with per-cell `argmax`, which preserves disjointness; measured cost **0 of 54 entities lost**. The original isolation test placed entities 32 px apart and never exercised it.

**Model E's result — a null, stated as one.** Sealed test set, 3,960 frames, identical data/split/seed/objective to A and C:

| | L1 | PSNR | SSIM | sharpness | gate |
|---|---|---|---|---|---|
| **A** (CNN) | **0.008947** | **34.041** | **0.9747** | **0.9104** | **PASS** |
| **C** (StructFormer) | 0.009174 | 33.858 | 0.9728 | 0.8514 | FAIL |
| **E** (entity-aware) | 0.009219 | 33.789 | 0.9726 | 0.8684 | FAIL |

- **E vs C: -0.49 % L1, +0.0170 sharpness.** Model C's measured across-seed spread is ~1.1 % (L1) and **sd 0.0079** (sharpness), so 2 sd is 0.0158. The L1 difference is inside noise; the sharpness gain sits *barely* outside it **on a single run**, which is a hint, not a result.
- **Entity conditioning produces no measurable change in image quality**, and does not close the gate (0.8684 vs 0.90). The CNN still wins outright on every metric at every dataset scale.
- **The direction is at least consistent with Decision 72**: committing to a per-entity appearance should sharpen, and E moved the right way. Whether that is real needs repeat seeds; it is **not** claimed here.
- **M7 the milestone still passes** — its exit criteria are correctness properties (Decision 74), all met. Model E's quality was never M7's bar. A null result at rung E is a finding, not a milestone failure.

**Consequence for M8.** Two independent signals now say the entity direction has thin material in this scene: memory carries no usable information at reappearance timescales (Decision 73), and entity conditioning without memory changes nothing measurable (here). Neither is a reason to abandon EPAM — it is the protected headline novelty — but M8's claims must be scoped to what can actually be demonstrated, and the metric it will be judged on should be fixed before it is built, not after.

---

### Decision 76: Entity Consistency Is Keyed by Identity, Not by Slot (2026-09-22)

Found by the M7 spec-guardian review, together with three smaller corrections to Decision 74. None of these affected any trained model — the consistency loss is wired into no training loop — but all were live defects in committed code.

**The defect.** `entity_consistency_loss` paired `tokens[:, k]` with `prev_tokens[:, k]`. Slot *k* is assigned by `select_entities`, which re-ranks **every frame** by `pixel_count`. Two entities swapping relative on-screen size — the ordinary case as objects approach and recede — swap slots, after which the loss pulled entity A's token toward **entity B's** previous token. That is identity resolved by **rank position**, the one thing Decision 9 says it must never be, and it made M7's "entity consistency loss" deliverable not an entity-consistency loss at all.

The function now takes per-frame `EntityKey` lists and pairs on them. Only entities present in both frames contribute; an entity that appeared, vanished, or was overflow-dropped has no pair, and pairing it would pull a real token toward a padded zero. A recycled `instance_id` has a different `generation`, so it is a different key and cannot inherit the destroyed entity's history — the same property M8's "destroyed entity cannot leak memory" depends on.

**Why the original tests missed it:** none of them swapped slots. The replacement asserts the swap case directly (A moves slot 1 → slot 0 with an unchanged token; correct loss is 0.0, the slot-keyed version scored 1.0), plus the mirror that genuine drift is still penalised — a loss that always returned zero would pass the first test alone.

**Three corrections to Decision 74, applied to that entry in place:**

1. **The Decision 66 citation was inverted.** It cited Decision 66 as the reason zero-init is the *remedy* for a never-engaging mechanism. The opposite is true: Decision 66 found Model B's **zero-initialised** fusion had never moved off zero, so zero-init is the mechanism that *failed* there. What caught it was measuring engagement **after training**, which rung E had not had at M7. The Model B risk was deferred, not addressed, and was resolved only when Model E trained (Decision 75).
2. **The TensorRT claim was overstated.** `m7_mask_strictness_trt.json` is FP32-only at `dim=32`, `18x32` — not Model E's bottleneck (448 wide, 36x64). Five ops it says "MUST be added at M7" were never added. **Model E has no export test**; the gap belongs to M9.
3. **The docstring said 1/16 resolution; it is 1/8** (`STEM_STRIDE=4` plus one halving = 36x64 at 288x512), so the export-safety argument was being made against 4x fewer attention positions than the model actually has. Two dead `hasattr` branches inferring the bottleneck width were also removed — neither attribute exists, so the code always fell through to the kwarg and was correct only because the fallback default happened to match.

**The M5-graph guard now fails closed.** It checked five hardcoded filenames by substring; a file added to `renderer/structformer/` later would have been silently unguarded while the guard kept passing. It now enumerates the package and inspects parsed `import` statements, so a name in a comment is not a false alarm and an aliased import is not a false pass. Verified by adding a file that imports the banned module and confirming the test fails.

---

### Decision 77: Entity Conditioning Sharpens — a Predicted Effect, Confirmed at 2.5 Sigma (2026-09-23)

Model E at three seeds (20260905 / 20260908 / 20260909 — the **same** seeds as Model C's variance runs, so the two spreads are measured under identical draws). Sealed test split, 3,960 frames. Evidence: `eval/reports/ladder_v3_e_seed_verdict.json`, generator `scripts/eval_ladder_v3_gates.py`.

| seed | L1 | sharpness | gate |
|---|---|---|---|
| 20260905 | 0.009219 | 0.8684 | FAIL |
| 20260908 | 0.009088 | 0.8720 | FAIL |
| 20260909 | 0.009153 | 0.8591 | FAIL |
| **E mean** | **0.009153** | **0.8665** (sd 0.0067) | |
| **C** | 0.009174 | **0.8514** (sd 0.0079) | FAIL |
| A (CNN) | 0.008947 | 0.9104 | **PASS** |

**E - C sharpness = +0.0151, pooled SE 0.0060 -> 2.5 sigma. Outside noise.**

- **This was predicted before it was measured.** Decision 72's averaging probe established the mechanism: the teacher flickers 32x the engine, training under `L1 + LPIPS` targets the **conditional mean** of that flicker, and averaging the same entity across consecutive teacher frames costs **6.2 % sharpness at 16.4 sigma**. Committing to one appearance per entity should therefore sharpen. It does. A pre-registered direction, confirmed on held-out sessions across three seeds, is materially stronger evidence than the same number found by inspection.
- **It gives EPAM a demonstrable mechanism**, which Decision 73 had removed. Memory across an occlusion carries no usable information here (residual correlation <= 0.037 at 30-480 ticks); **per-entity appearance commitment** does. M8's claim should be built on the second, not the first, and its metric is sharpness (pre-registered in Decision 72).
- **It does not close the gate.** No seed reaches 0.90; the mean recovers roughly a quarter of the distance from C to the CNN. **Model A still wins outright** on every metric at every dataset size.
- **L1 is unchanged** (-0.22 %, inside noise). The effect is specifically on sharpness, exactly as the averaging mechanism predicts — a broad quality gain would have been evidence *against* that explanation, not for it.
- **A mid-run reading was wrong and is corrected here.** While the seeds were training I noted that Model E looked *less* reproducible than Model C (val_l1 spread ~3.7 % vs ~1.1 %). That compared a mid-training value against a converged one. Final sharpness sd is **0.0067** for E and **0.0079** for C — E is slightly *tighter*, not looser.
- **Model E's single-run figure was the worst of its three seeds** (0.8684 vs a 0.8665 mean is above it, but its L1 0.009219 was the worst of the three). The original run understated the rung on L1 and roughly matched it on sharpness; Decision 75's "null result" verdict was correct for L1 and **wrong for sharpness**, which is why repeat seeds were run rather than assumed.

---

### Decision 78: M8 Design — EPAM Store, Model F, and a Built-In No-Memory Control (2026-09-23)

**Exit criteria (§10 M8), restated:** same entity retrieves same memory after leaving view; destroyed entity cannot leak memory to a later object; style change clears memory; EPAM improves over the non-memory model. Criteria 1-3 are lifecycle properties; the fourth is empirical.

**The store** (`renderer/epam/memory.py`) implements §4 verbatim: keyed by `EntityKey` (with `generation`), EMA update `M = gamma*M + (1-gamma)*F` only when `C > tau`, frozen while out of view, deleted on destroy, cleared on style change and session end, no eviction, keys from another session rejected. A lookup under a different style returns nothing even if a clear were missed. 17 tests assert each rule. **Criteria 1-3 are met** by that suite.

**Confidence `C_i`** is visible pixels over a "fully visible" budget of **1,600 px** — the *measured* median entity size (443 observations: p5 1,025 / p50 1,634 / p95 4,934). The first value, 800, was chosen without measuring and made the gate inert. Stated plainly: even at 1,600 the gate rarely fires in this scene, because entities here are large; it guards edge slivers, not typical frames.

**Model F** (`renderer/epam/model.py`) = Model E + memory + a learned per-style token table (live switching; Decision 15's cached encoding, Decision 70's scope). Resumes from Model E per the §4 curriculum: **all 111 tensors map, and F with no memory reproduces Model E with max difference 0.00e+00.** F adds 16 parameters (the second style row).

**One memory rule for training and runtime** (`renderer/epam/runtime.py::memory_step`) — a model trained under a subtly different rule than it runs under would pass every test and fail only in the game. It is keyed by `EntityKey`, never slot (Decision 76).

**The control is built into the code.** With `gamma = 0` memory reduces to the current descriptor — exactly Model E. So **F (gamma 0.9) and F0 (gamma 0)** train through identical code, clips, styles, objective, schedule, step count and selection rule, and **F - F0 measures memory and nothing else.** That is a cleaner ablation than F vs Model E, which trained for fewer steps, on one style, without clips. Both properties are tested: gamma = 0 is bit-identical to single-frame rendering, and gamma > 0 provably uses history.

**Clip training.** Memory is detached (§4), so history frames need only a no-grad forward: 4-frame clips, strides {1,2,4}, loss on the last frame. 67 % of validation clips have a last-frame entity also seen in history — the fraction on which memory can act at all.

**Style B: `watercolor`**, parameters copied unchanged from `oil_impasto` (only prompt and seed differ, so any alignment failure is attributable to the style). Stride probe: 90/90 frames pass, spurious max 0.149 vs a 0.35 gate, stylisation min 0.402 (stronger than oil's 0.287). Per Decision 40 the first full session is a hard gate; the chain aborts if it fails.

**Pre-registered expectation, recorded before training:** Decision 73 measured residual correlation <= 0.037 beyond 30 ticks, so memory across a *reappearance gap* should contribute little. Clips span 3-12 frames, where correlation is 0.14-0.72, so any F - F0 gain is expected at short lags. **Primary metric: test-set sharpness** (Decision 72); L1, RCS and temporal error reported alongside. A null F - F0 is an acceptable, reportable outcome.

---

### Decision 79: M8 Criterion 4 Not Met — EPAM Memory Does Not Improve Output (2026-09-26)

Model F (EPAM on, gamma 0.9) vs Model F0 (the built-in no-memory control, gamma 0), identical code, clips, styles, objective, schedule, step count and selection rule (Decision 78). Both resumed from Model E; both early-stopped at epoch 29 with best at epoch 17. Sealed test split, 3,912 clips, both styles. Evidence: `eval/reports/m8_f_vs_f0.json`, generator `scripts/eval_m8.py`. The pass threshold was written into the generator **before** the numbers existed: 2 sigma of run-to-run sharpness noise for one run each, `2*sqrt(2)*0.0073 = 0.0206`, the 0.0073 taken from the three-seed spreads of Decision 77.

| subset | n | F sharpness | F0 sharpness | F - F0 | F L1 | F0 L1 | L1 |
|---|---|---|---|---|---|---|---|
| **all** | 3912 | 0.8829 | 0.8956 | **-0.0127** | 0.009064 | 0.009158 | -1.03 % |
| oil_impasto | 1956 | 0.8775 | 0.8809 | -0.0034 | 0.009262 | 0.009291 | -0.31 % |
| watercolor | 1956 | 0.8882 | 0.9101 | -0.0219 | 0.008866 | 0.009025 | -1.77 % |
| memory can act | 2712 | 0.8634 | 0.8749 | -0.0114 | 0.010280 | 0.010360 | -0.77 % |

Entity-pixel sharpness: F 0.8034, F0 0.8209 (-0.0175, n = 2,732).

- **Criterion 4 ("EPAM improves over the non-memory model") is NOT met.** On the pre-registered primary metric the difference is inside seed noise, and its sign is **against** memory.
- **The direction is the informative part, and it matches the mechanism in reverse.** Decision 72 found the model's blur comes from averaging the teacher's per-frame flicker. An EMA over an entity's descriptors *is* an average across frames. So memory plausibly pulls the entity token toward the same mean that already blurs the output: slightly lower L1, slightly softer image, both inside noise. The pre-registered hypothesis in Decision 78 — memory commits to one appearance and sharpens — is **not supported**. Stated as the most consistent reading, not a measured mechanism: nothing here isolates *why* the sign is negative.
- **The L1 gain (-1.03 %) is not claimed.** Model E's three-seed L1 sd is ~0.72 %, so 2 sigma for one run each is ~2 %. It sits inside noise.
- **The watercolor subset (-0.0219) crosses the threshold on its own.** It is one of four subsets looked at, so a single crossing is what chance predicts and it is **not** reported as a separate finding.
- **Consistent with Decisions 53, 72 and 73.** Memory across a reappearance gap has no residual information to carry (correlation <= 0.037 beyond 30 ticks), and within a clip it adds averaging rather than commitment. Three independent measurements now say this scene gives EPAM nothing to recover.
- **No tuning was done after seeing these numbers.** A different gamma, or tau, would be a new pre-registered experiment with repeat seeds, not a rescue of this one.

**What M8 did deliver:**
- **Criteria 1-3 are met** — the lifecycle rules, asserted by 17 tests.
- **Live style switching works.** One model renders both styles through a per-style token. F0 reaches sharpness **0.9101 on watercolor**, above the 0.90 gate, the first StructFormer-family number to clear it on any style.
- **The ablation was clean.** Resume from E was exact (max diff 0.00e+00), the F and F0 starting validation errors agreed to 0.01 %, and gamma = 0 is bit-identical to single-frame rendering. So the null is a property of memory on this data, not of the setup.

**Status: M8 closes with criteria 1-3 met and criterion 4 a recorded negative result**, the same standing as M5: a finding about this data, reported as measured.

---

### Decision 80: Model B Was Never Trained With History — Decision 66 Retracted in Cause (2026-09-26)

**The defect.** Model B's ladder-v2 run went through `train_baseline`, whose data path yields single frames. `TemporalUNet.forward(x)` calls `fuse` only when `prev_features` is supplied, so **the fusion was never called during training and received no gradient for the entire run.** The v2 report shows it: `val_n_frames: 1980`, the single-frame validation set. The guard added for Model D (Decision 58) refused `temporal_structformer` in `train_baseline` but not `temporal_unet`, and that gap is exactly how B went through the wrong trainer.

**What this retracts.** Decision 66 measured B's history-on and history-off outputs as identical to six decimals — that measurement stands — and attributed it to "the zero-initialised fusion convolution never moved off zero". **The cause was wrong.** A zero-initialised layer that receives gradient moves; this one received none, because its input path was never exercised. B was never a temporal control, so:

- "B is Model A plus 524,800 parameters that do nothing" was true of that *checkpoint*, but it says nothing about CNN temporal conditioning;
- any B-vs-A or B-vs-D comparison drawn from ladder v2 is **void**;
- M6's verdict is unaffected — it rests on D vs C, and D trained correctly through `train_temporal`.

**Fix, verified before any GPU time was spent.** `train_temporal` now builds either temporal rung from `model.arch`, and records the arch actually trained in checkpoints and reports. `train_baseline`'s guard refuses both temporal archs. Through the correct trainer, B's `temporal_fuse` receives a **non-zero gradient at step 0** from real ladder-v3 history; through the old path it receives **none**. Both directions are asserted in `renderer/temporal/tests/test_model_b_trains_with_history.py`.

**Now running:** ladder-v3 B and D (`scripts/run_m9_bd.sh`), both through `train_temporal`, on the same data, seed, objective and selection rule as A/C/E.

---

### Decision 81: Entity-Anchored Teacher Noise Does Not Create Persistent Appearance — EPAM v2 Not Pursued (2026-09-26)

**Question.** Decision 79 found EPAM memory does not help. The most consistent explanation was that the training targets carry no persistent per-entity appearance for memory to recover: the teacher draws one noise field pinned to the **screen** (one generator per style), so a moving entity passes through different noise each frame. If giving each entity **its own** noise field — seeded from its `EntityKey`, mapped through its bounding box, restored when it reappears — produced a consistent per-entity look, an EPAM v2 would have a basis. This is distinct from Decision 60's refuted approach, which chained generated outputs; here no generated pixel feeds another frame.

**Probe** (`scripts/probe_anchored_noise.py`, `eval/reports/probe_anchored_noise.json`): one test session, `oil_impasto`, 10 reappearance sequences plus a 6-frame contiguous run, 46 frames each generated under both conditions with everything else identical. Run on CPU while the GPU trained. The noise override is verified to have taken effect: per-sequence differences are non-zero (-0.033 to +0.032) and flicker changed.

| source | RCS across reappearance | flicker (mean abs frame diff) |
|---|---|---|
| native engine | 0.9278 | 0.0014 |
| A: screen-pinned noise (today) | 0.9077 | 0.0082 |
| B: entity-anchored noise | 0.9077 | 0.0072 |

- **Reappearance consistency is unchanged:** B - A = **+0.0001, SE 0.0071, 0.0 sigma.** The per-entity look after a gap does not depend on which noise the entity carries.
- **Flicker drops 12 %** (0.0082 -> 0.0072) but stays **5x the engine**.
- **Alignment is unaffected:** 0 of 46 frames fail any gate under either condition.
- **Reading, not a measurement:** at `strength 0.5` img2img, the output is dominated by the init image and the depth ControlNet, both of which change with viewpoint. Noise is a minor contributor to per-entity appearance, so anchoring it cannot make an entity look the same from a different view. That is why the RCS gap to native is untouched.
- **Scale caveat:** 10 sequences, one session, one style. It is a probe that decides whether to build, not a population estimate; a small true effect below ~0.014 (2 SE) is not excluded. It does exclude an effect large enough to make memory worth rebuilding the teacher for.

**Decision.** EPAM v2 via teacher-side appearance anchoring is **not pursued.** Four independent measurements now agree that this data offers memory nothing to recover: no residual correlation across gaps (Decision 73), no decay with time unseen (Decisions 53/72), no gain from memory in the controlled F vs F0 ablation (Decision 79), and no way to create persistent appearance by anchoring the teacher's noise (here). A future EPAM test needs targets whose per-entity appearance is **not** determined by the current view — a temporally consistent (video) teacher or authored per-entity textures — which is outside this project's hardware and scope. The project proceeds to M9 and M10.

---

### Decision 82: The Temporal Trainer Had No Early Stopping; GPU Memory Headroom for the Display (2026-09-26)

**Early stopping.** Decision 62 requires one early-stopping rule for every rung, but it was only ever added to `train_baseline`. `train_temporal` — which trains Models B and D — ran every job to its epoch cap. This never changed a *selected* checkpoint (best.pt is always the lowest val_l1, the same rule for every rung), only wall time: Model D would have run all 60 epochs. `train_temporal` now applies the identical rule (patience 6 evaluations, min_rel_delta 0.001), and the ladder-v3 B and D configs carry the same `early_stopping` block as A/C/E.

**Model B (ladder v3) is complete and valid.** The run was stopped by the user at epoch 48 (22:00, 26 Sep) before it wrote its report. Its best.pt is from **epoch 33, val_l1 0.009234**, written by the run itself. Applying the Decision 62 rule to its logged history, early stopping would have fired at **epoch 45**. So the checkpoint is exactly the one a completed run would have selected. `eval/reports/ladder_v3_b_train.json` is rebuilt from the per-epoch log and says so.

**Display starvation.** At 22:00 the kernel logged `nvidia-drm ... Failed to allocate NVKMS memory` — training was holding ~7.5 GB of the 8 GB card, leaving the desktop unable to allocate. Model D now runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which returns freed segments instead of caching them. That drops D's footprint from ~7.4 GB to ~6.8 GB, with no display allocation errors since. This is an allocator setting: it does not change the computation.

**Throughput.** The temporal rungs now use 8 dataloader workers rather than 2. At 2 workers the GPU idled roughly half the time waiting on triples (three frames plus motion and mask per sample); at 8 it holds ~96 % utilisation, and B went from ~23 to 13.7 min/epoch. Worker count changes only throughput, not the samples, their order or the optimisation.

---

### Decision 83: M9 Image-Quality Ladder on One Scoring Path (2026-09-27)

All rungs trained on ladder-v3 data (20,790 frames) with the same seed, early-stopping and selection rule, and scored by `scripts/eval_m9.py` on the **same 3,960 sealed-test frames** (4 held-out sessions, `oil_impasto`). Each rung gets the inputs it actually runs with: B and D their real previous frame, E its engine masks. Evidence: `eval/reports/m9_ladder.json`.

| rung | L1 | PSNR | SSIM | LPIPS | sharpness | params |
|---|---|---|---|---|---|---|
| **A** U-Net | **0.008947** | **34.041** | **0.9747** | **0.0459** | **0.9122** | 7,853,763 |
| B U-Net + temporal | 0.009296 | 33.643 | 0.9734 | 0.0489 | 0.8973 | 8,378,563 |
| C StructFormer | 0.009174 | 33.858 | 0.9728 | 0.0466 | 0.8532 | 7,775,467 |
| D StructFormer + temporal | 0.009336 | 33.555 | 0.9727 | 0.0485 | 0.8577 | 8,236,747 |
| E StructFormer + entity | 0.009219 | 33.789 | 0.9726 | 0.0463 | 0.8837 | 7,053,547 |

- **A matches its earlier independent evaluation exactly** (L1 0.008947), so this path agrees with the previous ones.
- **Sharpness is aggregated here as a ratio of summed Sobel energies**, one definition for all rungs. It differs slightly from the per-report figures elsewhere (C 0.8532 here vs 0.8514 in `ladder_v3_test.json`). Compare within this table, not across reports.
- **The U-Net wins every image-quality metric** (M5, Decision 71, unchanged at this scale).
- **Temporal conditioning does not improve per-frame image quality.** B is **+3.9 % L1** behind A; D is **+1.8 % L1** behind C with equal sharpness. B-A is beyond the ~2 % two-run noise implied by Model E's three-seed L1 spread; D-C is at its edge. That is not the temporal rungs' purpose — their claim is **temporal stability** (M6: D cut temporal error 11.74 % vs C). Two caveats: B and D also carry the temporal loss term, so the gap mixes architecture with objective; and B is the **first** B ever trained with history (Decision 80).
- **Entity conditioning is the one StructFormer addition that helps:** E is +0.031 sharpness over C here on a single seed, consistent with Decision 77's three-seed +0.0151 at 2.5 sigma. It is the best transformer rung on sharpness and the smallest model in the table.
- **Single seed per rung, except E and C** (Decisions 67, 77). Differences under ~2 % L1 or ~0.02 sharpness are not claimed.
- F and F0 (M8) use the two-style clip protocol and are reported in Decision 79, not mixed into this table.

**Still to do for M9:** temporal error for every rung on this split — the metric B and D exist for, and the first real test of CNN temporal conditioning — and the naive raw-ID control N that §11 requires.

---

### Decision 84: Model A's TensorRT Failure Diagnosed and Fixed — GroupNorm's InstanceNormalization (2026-09-27)

Decision 48 recorded Model A's TensorRT build as **unresolved**: `Error Code 10: Could not find any implementation for ForeignNode[...]`, an NVRTC compilation failure inside a Myelin fused region, 4/4 reproductions. Diagnosed by changing one thing per variant on the ladder-v3 checkpoint (`scripts/diagnose_trt_model_a.py`, `eval/reports/m10_trt_model_a_diagnosis.json`):

| variant | builds |
|---|---|
| baseline | no |
| builder_optimization_level 0, 1, 2, 3, 4, 5 | no, at every level |
| nearest Resize rewritten as repeat_interleave | no |
| **GroupNorm rewritten as reshape / mean / var / affine** | **yes** |
| both rewrites | yes |

**Cause:** `nn.GroupNorm` exports as ONNX `InstanceNormalization`, and on this toolchain (TensorRT 11.2.1.2, CUDA 13.2, RTX 5060 Laptop) Myelin fuses it into a region its NVRTC compiler rejects. MaxPool and Resize were suspects and are cleared: the build fails with Resize removed and succeeds with it present. Why Model C, which also uses GroupNorm in its stem, builds is **not established** — plausibly a channel/shape combination that fuses differently — and is not claimed.

**Verified, not just built** (Decision 50: a green build is not evidence):
- The engine runs and its outputs are finite.
- With TF32 off, the max |TensorRT - PyTorch| over 5 random inputs is **4.26e-6**, far inside the 1e-3 correctness tolerance. The rewrite in PyTorch differs from the original by 2.1e-6, float rounding.
- With TF32 on (TensorRT's default), the difference is 4.0e-3. That is TF32 precision, not an operator error.
- The engine is 32.4 MB.
- **Latency is not yet claimed.** It measured median 16.4 ms / p95 21.3 ms FP32+TF32, **while another job shared the GPU**. It will be re-measured on an idle GPU before it is quoted against the budget.

**The fix is an export transform, not a model change:** `inference/export/trt_compat.py::for_tensorrt` returns a deep copy with every GroupNorm replaced by `ExplicitGroupNorm`, which reuses the original weight and bias. Training and checkpoints are untouched. Tests assert numerical equivalence, that the original model is unmodified, and that the exported graph contains no `InstanceNormalization`.

**Consequence:** Decision 48's "Model A's TensorRT build is unresolved in this environment" is **resolved**. The CNN — the best image-quality rung at every scale (Decisions 71, 83) — is deployable, which removes M10's largest known risk.

---

### Decision 85: Temporal Conditioning Cuts Flicker on Both Backbones — B Works Once Trained Correctly (2026-09-27)

Temporal error on ladder v3's sealed test split, every rung with its real inputs (history for B/D, engine masks for E), 501 moving pairs across 4 sessions (one session contributes only 6 — it is mostly static). Evidence: `eval/reports/m9_temporal_error.json`, generator `scripts/eval_temporal_error_v3.py`. The verdict rule was written into the generator **before** any number existed: a temporal rung improves only if its temporal error is lower than its no-history partner's by more than `2*sqrt(2)*sd_rel`, with `sd_rel` from Model E's three seeds.

| source | temporal error | vs partner |
|---|---|---|
| native engine | 0.000241 | |
| teacher targets | 0.006265 | |
| A U-Net | 0.003175 | |
| **B U-Net + temporal** | **0.002692** | **-15.21 %, IMPROVES** |
| C StructFormer | 0.004191 | |
| **D StructFormer + temporal** | **0.003741** | **-10.73 %, IMPROVES** |
| E seeds | 0.004172 / 0.004159 / 0.004264 | |

Noise: E seed sd **1.36 %** relative, so the threshold is **3.85 %**. Both temporal rungs clear it by a wide margin.

- **Model B is the steadiest renderer in the ladder.** Its first correctly-trained run (Decision 80) cuts flicker 15.2 % against A. The ladder-v2 verdict "B's fusion is inert" (Decision 66) is now **superseded, not just re-explained**: once B receives history in training, its temporal mechanism works — on the CNN too.
- **D replicates M6 on the larger dataset:** -10.73 % here vs -11.74 % on ladder v2 (Decision 67). This is a second, independent confirmation on different data.
- **The trade-off with Decision 83 is real and stated together:** B and D lose 1.8-3.9 % L1 on per-frame image quality while gaining 10.7-15.2 % temporal stability. For a live renderer, flicker is the more visible defect.
- **Objective caveat, not hidden:** B and D train with an additional temporal-consistency loss term. The gain belongs to "history + temporal loss", not to the fusion architecture alone. Separating the two would need a history-free rung trained with the same loss, which is not in the ladder.
- **Every model is still ~11-17x the engine's flicker** and bounded by the teacher's 26x. Temporal conditioning narrows the gap to the engine; it does not close it.

---

### Decision 86: M10 Live Loop — C++ Runtime, GDExtension, and a Live Capture Path (2026-09-27)

**What exists and is verified:**

- **The C++ TensorRT runtime** (`inference/cpp_runtime`, `structrender::Engine`). It links the same TensorRT 11.2.1 the Python stack uses, with headers fetched at the exact version. On a real test frame it reproduces PyTorch to **1.0e-6** (strict FP32) and **5.6e-4** (TF32). It never throws into the caller.
- **The `StructRenderer` GDExtension.** Inside headless Godot, on a recorded test frame, it matches PyTorch within **1 RGB8 level** (0 pixels off by more). Malformed input returns an empty array — the §4 fallback — and does not crash.
- **The live loop** (`game_project/tests/runtime/live_loop.gd`) runs the dataset's own training scene and exporter. 150 frames, **0 fallbacks**.

**The bottleneck was the exporter, not the model.** `export_frame()` is the recording path: 4 sequential render passes, then a per-pixel GDScript loop over 147,456 pixels calling `get_pixel` four times each, plus motion. It cost **292 ms/frame** (3.2 fps). It is left **untouched** — the dataset's determinism checks guard it.

**The live path is a separate, equivalent route:**
- `capture_live()` renders only the three passes a structure-only renderer reads (depth, normal, albedo), with the same camera sync and pass materials, and returns the raw Images.
- `StructRenderer.render_images()` decodes them in C++, reproducing `export_frame()`'s rule exactly (depth <= 0 is background: depth +inf, normal and albedo 0).
- The pass images are **`FORMAT_RGBH`** (3 half-float channels). This was measured after the first implementation assumed RGBA and refused them; the reader names the formats it accepts and refuses anything else.

**Equivalence is proven, not argued:** on 3 frames, the live path and the recording path produce **byte-identical** output (max byte diff 0). The live renderer sees exactly the input distribution the model was trained on.

| | recording path | live path |
|---|---|---|
| structure export | 292 ms | **23.3 ms** |
| decode + inference + upload | 23 ms | 25.1 ms |
| total | 316 ms (3.2 fps) | **48.5 ms (20.6 fps)** |

Measured **while Model N trained at 99 % GPU**, so these are upper bounds; they are re-measured on an idle GPU before being compared to the §10 target.

**Remaining for M10's exit criteria** (~30 fps; standalone TensorRT <= ~20 ms):
- Render the three passes in parallel SubViewports in one frame, not sequentially (each sequential pass waits ~3 engine frames).
- An FP16 engine.
- Idle-GPU measurements.
- Live style switching needs the two-style model (F/F0) exported with its style and mask inputs.

---

### Decision 87: A Training Dataset From the Playable Game (2026-09-27; written 2026-09-28)

> **Written retroactively.** `game_project/tests/dataset/game_record.gd` and commit `8128ac7` cited this entry as "pending" from 2026-09-27; it is entered now.

**Why.** Every ladder model was trained on a scene the seed recorder builds in code. The playable arena collector (`main.tscn`: player, coins, spawned enemies) was never seen, so no model can render the actual game. The M10 live demo therefore showed the recorder's scene, not the game.

**How — no game code modified.** `game_record.gd` loads `main.tscn` and records it through the **same** exporter, adapter and recorder as the dataset, so the output has the same schema and passes the same M2 validator.
- The game's own `SimDriver` is kept out through the `structrender_no_sim` meta it already honours.
- The player is steered by pressing the game's own input actions, with a seeded wandering route per session variant (start position, random arena points, nearest coin).
- Each recorded frame advances the game by **exactly one physics step** (`max_physics_steps_per_frame = 1`, then unpause -> physics step -> pause). The tree stays paused while the exporter renders its passes, so depth, normal, albedo, ID and native RGB all show the same instant. Without the pause, the player and enemies move between passes and the buffers disagree.
- **Found in testing:** the game's spawner adds enemies to `get_tree().current_scene`, which is null when the scene is loaded by script. The first test run recorded a game in which no enemy spawned. `current_scene` is now set.

**Result.** **24 of 24 sessions pass the M2 validator** (990 frames each). A 240-frame test session showed 7 entities with spawn, visible, hidden and destroy events: coins collected, enemies spawned. The scene is flat, untextured grey geometry, as `main.tscn` is authored. The session list is `configs/data/game_sessions_v1.txt`. The split `configs/data/game_split_v1.json` was fixed by position before any target, model or metric existed: **18 train (17,820 frames) / 3 val / 3 test.**

**A regression this caused, and its fix.** The sessions were recorded into `data_pipeline/recorder/`. The validator and teacher test fixtures chose their subject as "the newest ds_* session", so they silently moved onto a gameplay session.
- The validator suite failed with **38 errors**: a gameplay session has no reappearance sequences.
- The teacher suite kept passing, but against a gameplay session with no teacher targets — a guard that no longer guarded.

This is the Decision 65 failure class again, and `main` carried it from `1199520` until it was fixed. Both suites now read one pinned session (`data_pipeline/seed_session.py`, `ds_1788687350_337657ea`, the committed seed session of Decisions 38-40). They **skip with a reason** if it is absent, rather than falling back to another session. A regression test plants a newer session directory and asserts the subject does not change.

---

### Decision 88: Selecting the Game Teacher Style by the Alignment Gates (2026-09-28)

**Status: designed, NOT yet run.** No result exists; this entry fixes the selection rule before any number does.

**Why.** The ladder's teacher runs at img2img strength 0.50, chosen in Decision 32 because stronger settings invented geometry. The resulting style is light enough to be hard to see (`eval/showcase/comparison_test_frame.png`). The game's flat grey geometry may leave room for a more visible style, but how far to go is decided by the four alignment gates, not by taste.

**Rule, pre-registered** (`scripts/probe_game_style.py`):
- Grid: `{oil_impasto, watercolor}` x strength `{0.50, 0.60, 0.70}`. Everything else is as committed.
- Scored on a stride-22 subset (45 frames) of the first game **train** session, measuring:
  - per-frame gate failures;
  - stylisation strength as colour-histogram travel from the native frame.
- **Pick the strongest setting with zero gate failures.** It must then pass a **full 990-frame session**, because Decision 40 recorded `cel_bold` passing an 81-frame probe and failing 36/990.
- If no setting has zero failures, the committed strength-0.50 styles are used.

---

### Decision 89: The Naive Raw-ID Control Matches Structured Entity Use on This Scene (2026-09-28)

Model N (§11 row N; Decision 9's "Allowed Control") is Model C plus one input channel, `entity_id / 8`, at 1.0003x C's parameters. It was trained with the same data, seed, objective, early stopping and selection rule, and scored on the same 3,960 sealed-test frames by `scripts/eval_m9.py`. Evidence: `eval/reports/m9_ladder.json`, `eval/logs/ladder_v3_n/`, `eval/logs/m9_ladder_with_n.log`.

| rung | L1 | PSNR | LPIPS | sharpness | vs C sharpness |
|---|---|---|---|---|---|
| C StructFormer | 0.009174 | 33.858 | 0.0466 | 0.8532 | |
| **E** structured (masked lookup) | 0.009219 | 33.789 | 0.0463 | **0.8837** | +0.0305 |
| **N** raw ID as a number | **0.009054** | **33.865** | **0.0459** | **0.8796** | +0.0264 |

- **The control does NOT fail the way §11 intends.** N captures about 87 % of E's sharpness gain and has *lower* L1 than both C and E. N-E sharpness is **-0.0041**, far inside the measured seed spread (E sd 0.0067). N and E are **indistinguishable** on this scene.
- **Predicted before training, as the reason to read it this way.** Decision 83's build note and `renderer/controls/raw_id.py` recorded that only IDs 0, 1 and 2 occur in this data. That makes the "naive" channel nearly a clean two-entity label: "a strong version of the naive control". With so few, stable IDs, a numeric channel carries the same information as masked addressing.
- **What this does and does not show.**
  - It shows that **on this scene, exposing IDs is enough**. The entity sharpness gain of Decision 77 is real, but it does not require structured use to obtain.
  - It does **not** show that structured use is unnecessary in general. The failure modes the raw channel should hit — recycled IDs, many entities, ID values with no meaning in magnitude — are **absent from this data**, so the control could not exercise them.
  - Separating the two needs a scene with many entities and ID recycling. That is not in scope.
- **Single seed each for C and N**; E has three. Differences under ~0.015 sharpness are not claimed.
- **Decision 9 is unchanged.** The production rule (IDs are addresses, never numeric channels) rests on generalisation and lifecycle safety — a recycled `instance_id` must not look like the entity it replaced — not on this scene's image metrics. N remains a control, isolated in `renderer/controls/`, and asserted by tests to be imported by no production module.

---

### Decision 90: Game Teacher Targets — Oil Passes, Watercolour Does Not (2026-09-29)

Decision 88's rule applied end to end. Evidence: `eval/logs/game_style_probe.log`, `eval/logs/game_style_fullgate.log`, `eval/logs/game_teacher_chain.log`, `eval/reports/game_style_probe.json`.

**The 45-frame probe** (first game train session, stride 22):

| style | 0.50 | 0.60 | 0.70 |
|---|---|---|---|
| oil_impasto | **0/45**, stylisation 0.754 | 1/45 | 9/45 |
| watercolor | 1/45, 0.441 | 0/45, 0.523 | 1/45 |

**Full-session gates (990 frames), as the rule requires:**
- **oil_impasto 0.50: PASS, 0/990.**
- **watercolor 0.60: FAIL, 5/990** — 2 spurious-edge, 2 not-stylised, 1 silhouette. It passed the 45-frame probe, then failed the full session: exactly the Decision 40 pattern the full-session requirement exists to catch.
- **watercolor 0.50** (the committed setting, all 29 seed-scene sessions ALIGNED): **FAIL, 10/990** — 9 not-stylised, 1 boundary. On the game's flat grey geometry, watercolour at 0.50 often barely changes the frame.

**No watercolour setting is usable on the game.** Consequence: **a two-style game model (live oil <-> watercolour switching, F0) cannot be trained on data that passes the gates.** Not trained around; recorded.

**Oil generated for all 24 game sessions: 23 ALIGNED, 1 FAIL.** `ds_1790535959_640a0d22` fails on a single frame (frame 30, entity 4, silhouette support 0.511 < 0.6). It is a **train** session, so it is **dropped whole**, as Decision 69 did with a bad session. `configs/data/game_split_v1_oil.json`: **17 train (16,830 frames) / 3 val / 3 test**. Val and test are the ones fixed before any target existed, and are untouched.

**Stylisation strength on the game:** oil at 0.50 reaches a mean stylisation of 0.756 on the full session, about 2.5x the seed scene's at the same setting. The flat grey geometry leaves the teacher more to paint, which answers Decision 88's motivation without raising strength.

---

### Decision 91: Model B on the Playable Game — Held-Out Results (2026-09-30)

Model B (U-Net + temporal fusion, 8,378,563 params) trained from scratch on the playable game: `oil_impasto` 0.50, `configs/data/game_split_v1_oil.json` (17 train sessions / 16,830 frames), `configs/game/model_b_oil.json`. The first launch ran out of GPU memory because desktop applications held ~1.9 GB; it was relaunched at batch 4 x accumulation 2 (effective 8, Decision 47). Trained 11.3 h and early-stopped at epoch 51; best epoch 39, val L1 0.006442. Started with the user's explicit go-ahead. Evidence: `eval/reports/game_b_oil_train.json`, `eval/logs/game_b_oil/`.

**Sealed test sessions** (3, fixed by position before any target existed; 2,970 frames, model rendered with its real previous frame), from `scripts/eval_game.py` -> `eval/reports/game_b_oil_test.json`:

| metric | value |
|---|---|
| L1 | 0.00606 |
| PSNR | 35.43 dB |
| SSIM | 0.9814 |
| LPIPS | 0.0216 |
| **sharpness ratio** | **0.9465** (gate 0.90) |

| temporal error (2,455 moving pairs) | |
|---|---|
| native engine | 0.000100 |
| teacher targets | 0.010416 |
| **Model B** | **0.005582** — 46 % below its own supervision, 56x the engine |

- **First rung to clear the 0.90 sharpness gate by a wide margin on held-out data.** On the seed scene the best rung was A at 0.912.
- **Not an architecture comparison.** The scene differs (flat grey geometry with far fewer edges), and the teacher styles it ~2.5x more strongly (Decision 90). These numbers compare with nothing in the ladder. Only one rung was trained on the game; there is no A/C/E on this data, and no seed repeats.
- **The same pattern as Decision 85:** the model is steadier than the teacher it learns from, and still far from the engine.
- **Oil only.** No watercolour passes the gates on this game (Decision 90), so this model cannot switch styles.

---

### Decision 92: A Bold Watercolour Passes on the Game at Strength 0.55 (2026-09-30)

The committed watercolour failed the game's full-session gate (Decision 90), mostly as "not stylised": pale washes over large flat grey surfaces barely change the frame (`eval/showcase/watercolor_failure_frame440.png`). This is a **prompt** change, not a threshold change — the gates are exactly as committed. `configs/teacher/watercolor_bold*.json` asks for saturated pigment, a tinted paper ground and colour blooms across every surface, with seed 20260930 and other parameters as committed.

**Probe** (45 frames, first game train session; `eval/reports/game_style_probe_watercolor_bold.json`):

| strength | failing | stylisation |
|---|---|---|
| 0.50 | 0/45 | 0.822 |
| 0.55 | 0/45 | 0.912 |
| 0.60 | 0/45 | 0.972 |

For comparison, the old watercolour at 0.50 scored 1/45 and 0.441 — the bold prompt changes the frame about twice as much.

**Full-session gate** (990 frames, strongest first, stopping at the first pass; `eval/logs/game_style_fullgate_bold.log`):
- **0.60: FAIL, 2/990** — spurious edges at frames 433-434, the same floor-dominated view that defeated the old watercolour. A 45-frame probe again missed what the full session found (the Decision 40 pattern).
- **0.55: PASS, 0/990.**

**Consequence:** oil 0.50 (Decision 90) plus **watercolour_bold 0.55** gives two gate-passing styles on the game, so a two-style game model with live oil <-> watercolour switching is possible again. It needs watercolour_bold_s055 targets for the remaining 23 game sessions (one is done) and a two-style model trained on both. Neither has been started. Training requires the user's explicit go-ahead.

---

### Decision 93: Bold Watercolour Across All Game Sessions — 17/24 ALIGNED (2026-09-30)

`watercolor_bold_s055` was generated for all 24 game sessions (23 new plus the gate session), 11:09 -> 19:59. Evidence: `eval/logs/game_teacher_watercolor_bold.log`. No training was started.

**17 of 24 sessions ALIGNED; 7 FAIL, each on only 1-2 of 990 frames:**
- spurious-edge failures in 4 sessions;
- boundary-F failures in 3 sessions (one of them — val `81203cd9` — at 2 frames);
- one entity-silhouette failure.

The Decision 92 full-session pass on one session did **not** generalise: roughly 30 % of sessions fail somewhere. This is the Decision 40 pattern again, now at the level of sessions rather than frames.

**Sessions with BOTH styles ALIGNED** (what a two-style model can use under the whole-session rule of Decisions 69 and 90):

| split | both styles | lost to watercolour |
|---|---|---|
| train | 12 / 17 | 6d509a33, 716f96e0, 62b8325b, 9fbff551, 85c67b3e |
| val | 2 / 3 | 81203cd9 |
| **test** | **2 / 3** | **4815ad01** |

- **The held-out test set is hit.** The test sessions were fixed by position before any target existed, so losing one cannot be repaired by substitution — only by shrinking the test set (1,980 frames instead of 2,970) or by changing the exclusion policy. Neither is decided here.
- **Model B (oil only, Decision 91) is unaffected:** its data and its held-out evaluation are complete.
- **An open policy question for the user, not decided here.** Every failure is 1-2 frames out of 990, and the whole-session rule discards the other ~988 gate-passing frames with it. A per-frame exclusion rule would keep every frame that passes every gate, so no failing frame is ever used. That would retain all 17 train and all 3 val/test sessions, but it is a change of **data policy**, and it needs its own Decision entry if adopted. It is not a threshold change: the gates stay exactly as committed.

### Decision 94: Per-Frame Exclusion for the Two-Style Game Model; Two-Style Model B Built (2026-09-30)

**Data policy change, adopted by the user after seeing Decision 93's failures.** This is disclosed plainly: the rule was chosen knowing which sessions it would bring back. What makes it acceptable is what it does *not* change. The four alignment gates and their thresholds are exactly as committed; no frame that fails any gate is ever trained on or scored; the test sessions are the ones fixed by position in `game_split_v1.json` before any target existed.

**Rule.** A frame that fails ANY alignment gate in EITHER style is excluded from BOTH styles, together with every temporal triple or history pair that touches it. A frame whose predecessor is excluded is scored with no history (zero mask, the existing cold-start path). The unit of exclusion changes from session (Decisions 69 and 90) to frame. Nothing else changes.

**How the list is built:** `scripts/build_frame_exclusions.py` -> `configs/data/game_frame_exclusions_v1.json`, log `eval/logs/game_frame_exclusions.log`.
- It runs the M3 validator **itself** (`validate_alignment`, default `Thresholds()`, `write_marker=False`), rather than a re-implementation of its gate loop.
- A (session, style) without an ALIGNED marker is **admitted only if every validator error is a per-frame `align.*` gate failure**. Layout, header, frame lines and frame count must all be clean, and the target set must be complete. Per-frame exclusion can remove a bad frame; it cannot paper over a broken target set.
- **Result: 11 frames excluded out of 23,760 (0.05 %). All 24 sessions are admitted in both styles, and none is refused.** The recomputed failures are identical to the failures logged when the targets were generated: oil 1/1, watercolour 10/10.

| split | frames per style | triples per style |
|---|---|---|
| train (18 sessions) | 17,820 -> 17,813 | 17,784 -> 17,763 |
| val (3) | 2,970 -> 2,968 | 2,964 -> 2,958 |
| test (3) | 2,970 -> 2,968 | 2,964 -> 2,958 |

The full pre-registered 18/3/3 split is restored for the two-style model. That includes oil session `640a0d22` (Decision 90), which loses 1 frame instead of all 990, and test session `4815ad01` (Decision 93), which loses 2.

**Known bias, accepted.** The excluded frames are the ones on which the teacher was hardest to keep aligned: busy views with many edges, and thin silhouettes. Removing them under-represents those views slightly, in training AND in the test score. At 11 frames in 23,760, the effect on any mean is far below the noise yardsticks. It is recorded so a later reader does not have to rediscover it.

**Model B (oil, Decision 91) is not re-scored or changed.** Its numbers stand under the whole-session rule it was trained under.

**Two-style Model B — architecture.** `renderer/temporal/style_unet.py::StyleTemporalUNet` is Model B's TemporalUNet with three changes:
- **Style-free encoder.** The encoder reads only engine structure, so the history features are identical whichever style is on screen. A live style switch cannot disturb the temporal memory.
- **Per-style FiLM on every decoder level.** A per-channel scale and shift comes from a learned style table (16-d, one row per trained style), so style reaches colour and texture at every scale.
- **Zero-initialised FiLM projections.** An untrained two-style B is exactly Model B, which is tested bit-for-bit for both styles, and the difference between styles is learned. The style table is trunc-normal rather than zero, so the FiLM weights receive gradient at step 0.

Overhead: +16,352 parameters on 8.38 M (+0.2 %). Styles are a fixed, trained set chosen by index; there are no zero-shot styles (Decision 70). EPAM is untouched.

**Training setup (not started — needs the user's explicit go-ahead).** Config: `configs/game/model_b_two_style.json`. It trains on every (session, style), with the objective, early stopping, batch 4 x accum 2 and workers identical to game B oil. Checkpoint selection is on `val_l1` averaged over both styles' val frames, and both styles have the same frame count. Plumbing:
- `train_temporal` passes `style_idx` through `training_forward` and `eval_forward`.
- `train_baseline` refuses the arch and registers it for `load_checkpoint`.

Tests: `renderer/temporal/tests/test_two_style_b.py`.

**Still to do after training:** `scripts/eval_game.py` needs a style argument and the exclusion list before it can score this model, and the scoring must apply the same exclusions.

### Decision 95: Two-Style Model B on the Game — Held-Out Test Results (2026-10-02)

**Training.** Trained under Decision 94, on the user's explicit go-ahead for this run. It ran all 60 epochs (no early stop) in 29.2 h, with the best checkpoint at epoch 49 (val_l1 0.007435, over both styles). Evidence:
- `eval/logs/game_b_two_style/train.log`
- `eval/reports/game_b_two_style_train.json`
- checkpoint `checkpoints/game_b_two_style/best.pt`

**Test scoring.** The sealed test sessions (4815ad01, f1f9c484, 8d71c460) were scored once with `scripts/eval_game.py`, which gained a style argument and the Decision 94 exclusion list. 2,968 frames per style; flicker (temporal error) on 2,451 moving pairs. Log: `eval/logs/game_b_two_style/test_scoring.log`.

Oil-only B was **re-scored on the same 2,968 frames**, so the comparison is like-for-like. That report is `game_b_oil_test_d94frames.json`. Its Decision 91 numbers in `game_b_oil_test.json` are unchanged: removing the 2 frames moved nothing at 4 significant figures.

| model, style | L1 | PSNR | SSIM | LPIPS | sharpness | flicker: model / teacher |
|---|---|---|---|---|---|---|
| oil-only B, oil | 0.00606 | 35.43 | 0.9814 | 0.0216 | 0.946 | 0.00558 / 0.01041 |
| **two-style B, oil** | 0.00610 | 35.33 | 0.9807 | 0.0225 | 0.946 | 0.00566 / 0.01041 |
| **two-style B, watercolour** | 0.00805 | 34.79 | 0.9786 | 0.0243 | 0.937 | 0.00614 / 0.01270 |

**Findings:**
- **One model now renders both styles.** Both clear the 0.9 sharpness floor. In both, flicker is about half the teacher's (0.54x oil, 0.48x watercolour).
- **Sharing costs oil almost nothing:** PSNR -0.10 dB, LPIPS +0.0009, sharpness identical, flicker +1.4 %. Whether even that small cost is real cannot be said. There is one run of each model and no game-level seed variance has been measured, so it is reported as a measurement, not as a significant difference.
- **The style switch is real, not a blend.** On test session f1f9c484 (99 frames, every 10th), each style's output was compared with each style's targets:

  | output | vs oil targets | vs watercolour targets |
  |---|---|---|
  | oil style | 0.0058 | 0.0257 |
  | watercolour style | 0.0247 | 0.0079 |

  The two outputs differ from each other by L1 0.0242, against 0.0260 between the two teachers' images. The model reproduces about 93 % of the inter-style gap from a 16-d style vector, while the encoder and temporal memory are shared and style-free.

This is a sampled check, not a full-set metric; the numbers are in this entry only.

**What this does not show.** Colour-histogram travel ("style acquisition", the M4 criterion) was not computed by this scorer. The live in-game switch has not yet been exported or run; that is the next step (M10 path: ONNX -> TensorRT -> GDExtension).

### Decision 96: Live Style Switching in the Playable Game — Working, Without Frame History (2026-10-02)

**What runs.** `game_project/tests/runtime/live_game.gd` plays `main.tscn` normally (WASD/arrows + Space). Every frame it is captured with `StructuralExporter.capture_live()`, rendered by two-style B through the C++ runtime and GDExtension, and shown full-screen. Keys: **1** oil, **2** watercolour, **0** native.

**One engine per style.** `scripts/m10_two_style_engines.py` builds both from the ONE two-style checkpoint: each engine is the model with its style index fixed as a graph constant. Every engine is therefore 1-input/1-output, the contract the runtime already had (Decision 86), so no C++ change was needed. Both engines are loaded once; switching style just chooses which engine renders the next frame.

**C++ parity, both styles.** On a real game test frame (`eval/reports/m10_two_style_engines.json`):
- **Strict FP32:** max |C++ - PyTorch| = 1.4e-06 (oil) and 1.4e-06 (watercolour); PASS at 1e-3.
- **TF32 (what ships):** about 1.1e-3.
- **The styles differ through C++:** mean |oil - watercolour| = 0.0161.

**Live check.** `SR_FRAMES=300`: a scripted walk through the game, cycling oil -> watercolour -> native every 100 frames. Log: `eval/logs/live_game_check.log`; screenshots: `eval/reports/live_game_check/`.
- Median frame time **19.9 ms (50 fps) for oil, 21.9 ms (46 fps) for watercolour**, including capture, inference and display. That is above the 30 fps M10 target.
- **0 fallbacks** in 300 frames.

**The cost — stated plainly.** These engines run **without frame history**: every frame is a cold start, because the live loop does not yet supply the previous frame's features, motion or confidence mask. Scored on the same test frames as Decision 95 (`game_b_two_style_test_*_nohistory.json`):

| style | PSNR with -> without history | LPIPS | sharpness | flicker |
|---|---|---|---|---|
| oil | 35.33 -> 33.58 | 0.0225 -> 0.0334 | 0.946 -> 0.930 | 0.00566 -> 0.00509 |
| watercolour | 34.79 -> 32.82 | 0.0243 -> 0.0359 | 0.937 -> 0.918 | 0.00614 -> 0.00571 |

- **History is worth about 1.8-2.0 dB and about 33 % of LPIPS.** The model was trained almost entirely with history, so a cold start is off its training distribution.
- **Flicker does not get worse without history.** It is slightly lower, consistent with the cold-start output being a little softer.
- **The live demo is therefore below the Decision 95 numbers.** Matching them needs the temporal live path:
  - the previous frame's bottleneck features fed back;
  - the engine motion pass;
  - the confidence mask (depth/normal reprojection plus spawned-entity masks) computed per frame in the runtime.

**Display notes.**
- The model renders 512x288 (16:9). With a non-16:9 window the native view (0) is cropped differently, so it frames wider or narrower than the styled views. This is a display difference, not a model one.
- The styles are deliberately subtle. The game is grey with plain materials, and the teacher strengths that pass the alignment gates are about 0.5 (Decisions 90, 92), so the targets themselves are mild.

### Decision 97: Live Temporal Path — the Playable Game Now Renders With the Scored Model's History (2026-10-02)

Closes the gap Decision 96 left open: the live game ran two-style B without frame history, about 2 dB below its scored quality. **The live path now supplies exactly the inputs the scored model was evaluated with.** No training was involved.

**Graph.** `renderer/temporal/live_step.py::LiveStep` is one exported step per style:
- **Six inputs:** x_t, x_prev, feat_prev, motion, no_prev (spawned-entity pixels), motion_valid.
- **Two outputs:** the frame, and this frame's encoder features, which are fed back as next frame's `feat_prev`. That is identical to `encode(structure_prev)`, which `eval_forward` recomputes, so the encoder is not paid for twice.
- **The confidence mask M_t is built inside the graph** with the same `confidence_mask` the training data and the scorer use, so no second implementation can drift. Its depth, finite flag and normals are recovered from the structure channels (depth = 1/x0 - 1; the round trip is about 1e-6 m against the 0.175 m mask scale).

**C++.** New `StructRenderer` methods; the single-frame methods are untouched:
- `load_temporal_engines` / `set_style` / `reset_history` / `render_temporal`;
- `StructuralExporter.capture_live_temporal()` adds the entity-id pass, the camera and the entity transforms. `export_frame` and `capture_live` are untouched.

Motion and the spawned-entity mask are ported line for line from `_compute_motion`, using Godot's own `Transform3D` / `Projection` types. All styles share one history, which is valid because the encoder is style-free (Decision 94): a style switch keeps it. Native mode resets it.

**Verification — every link in the chain:**
1. **LiveStep == the scorer.** On real test frames in both styles, max |diff| vs `eval_forward` is 2.4e-6 on RGB and 2.1e-5 on M_t vs `engine_confidence`. Unit tests: `renderer/temporal/tests/test_live_step.py`; with no history the step equals the cold-start model exactly.
2. **TensorRT == PyTorch** (`eval/reports/m10_live_temporal_engines.json`, real game test pair): strict FP32 max |diff| is 1.6e-06 on RGB and 7.3e-06 on the fed-back features.
   - Engine time: **6.15 ms strict FP32 vs 9.37 ms TF32**. Strict is both exact and faster on this GPU, so **strict ships**.
3. **C++ motion == GDScript motion** (`game_project/tests/runtime/live_temporal_parity.gd`, log `eval/logs/m10_live_temporal_parity.log`): 120 lockstep frames of the playable game, with `export_frame` and the live path on the same instant and the style alternating every frame. Max |motion difference| = 7.5e-8 UV; motion_valid agreed on all 120; 0 failures.
4. **Live C++ frame == scored model** (`scripts/m10_live_temporal_parity.py`, `eval/reports/m10_live_temporal_parity.json`). The last two parity frames are rebuilt in PyTorch from the dumped structure and motion: **max difference 1/255 (rounding), 0 pixels off by more.**

Together these mean the live frame IS the Decision 95 model's frame for the same game state.

**Not exercised: the spawned-entity branch (KB S7).** The game spawns its one enemy on frame 1, before the first captured pair, so `no_prev` was 0 pixels throughout. The C++ branch mirrors `_compute_motion`'s skip exactly and the in-graph mask's handling is unit-tested, but it has not been seen live.

**Live check** (`SR_FRAMES=300`; log `eval/logs/live_game_temporal_check.log`; screenshots in `eval/reports/live_game_temporal_check/`): **42.6 fps oil, 43.3 fps watercolour** (median 23.1-23.5 ms per frame, including the 4-pass capture, motion, inference and display); **0 fallbacks**. That is about 7 fps below the no-history path because of the entity-id pass and the motion computation, and still above the 30 fps M10 target.

**Free play is not lockstep.** The tree is not paused during the 4-pass capture, so a physics tick can occasionally fall between passes. The parity test ran lockstep, as recording does. This is a deliberate choice for real-time gameplay; any artefact from it is limited to moving entities on the frames where it happens.

### Decision 98: M10 Closed — FP16 Engine, Idle Measurements, Failure Injection, Runtime Log (2026-10-02)

**M10 exit criteria (§10), each checked explicitly.** All runs are on the playable game, with two-style B live with history (Decision 97), the GPU otherwise idle (0 % utilisation before each run; only desktop GPU processes present), and the window on the real display. Evidence: `eval/logs/m10_final/`.

| criterion | result | evidence |
|---|---|---|
| stable live loop | **3,000 frames, 0 fallbacks.** Median styled frame 22.4 ms in the first half vs 22.3 ms in the second (no drift); p95 24.5 ms, p99 25.2 ms. The only frame over 38 ms is frame 0 (131 ms, warm-up). Gameplay advanced 2,755 physics frames. | `stability_fp16.log`, `stability_fp16_frames.csv` |
| ~30 fps | **44.6 fps oil, 44.7 fps watercolour** (FP16); strict FP32 41.8 / 41.0 fps | `stability_fp16.log`, `fp32_strict_live.log` |
| TensorRT standalone <= ~20 ms | **4.53 ms FP16, 6.23 ms strict FP32** per live step (median of 500, idle GPU) | `eval/reports/m10_fp16_live.json` |
| renderer failure does not break gameplay | see the failure tests below | `failure_injection*.{log,csv}`, `no_engine.log` |

**Failure tests:**
- **Injected failure.** Every 50th styled frame's render is made to fail on purpose (no entity-id pass). All 8 failures fell back to the native frame; the game kept running (561 physics frames, 14.5 m travelled). Each failure was followed by a successful render on the next styled frame, which starts with no history (`motion_valid` 0); history resumes on the frame after. 7 of 8 are observable in the log; the 8th was followed only by native-mode frames until the run ended.
- **Missing engine.** The game runs native-only and logs the error.

**FP16 engine (deliverable).** Built per Decision 45: an FP16 ONNX graph in a STRONGLY_TYPED network. The confidence mask stays FP32, because depth in metres needs it; engine I/O stays float32. The FP16 tolerance Decision 45 deferred to M10 was **fixed in `scripts/m10_fp16_live.py` before any FP16 number existed**. It is measured on 99 real test pairs per style against the strict-FP32 engine:

| criterion | limit | oil | watercolour |
|---|---|---|---|
| mean \|dPSNR\| vs teacher | < 0.05 dB | 0.044 dB | 0.039 dB |
| mean \|pixel difference\| | < 1/255 | 0.0004 | 0.0004 |

**PASS in both styles** — not by a wide margin on dPSNR (0.044 against 0.05).

**What the tolerance does not cover, stated.** The **maximum** single-pixel difference is 0.41 (oil) / 0.52 (watercolour), and the worst single frame moves 0.74 dB. Investigated: 16 of 99 frames have any pixel off by more than 0.1, at most about 200 pixels per frame, and **all lie on one-pixel object outlines**. FP16 shifts an edge by a fraction of a pixel, and an edge pixel is either surface or background. Side by side, the frames are indistinguishable. This was found after the tolerance was set; it does not change the verdict, and it is recorded so the "PASS" is not read as "pixel-identical".

**Shipping.** The pre-registered rule was: ship FP16 if it passes AND is faster. It passes, and it is 27 % faster standalone and about 3 fps faster live, so **FP16 ships**. The live-equivalence chain of Decision 97 was proven on strict FP32. FP16 inherits it within the tolerance above, not bit-for-bit.

**Runtime log (deliverable).** `live_game.gd` with `SR_LOG=path` writes one CSV line per frame: frame, style, ms, rendered, injected, motion_valid, error.

**Decision 86's other open item** — rendering the passes in parallel SubViewports — was not needed: the target is met with sequential passes.

**Verdict: M10's exit criteria are met** on this hardware, at the development resolution (512x288).

---

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

Notes carried from M1:

- The M1 exporter's own guard (`frame_structure.md` §6) catches *total* channel
  failure (a whole pass returning empty / all-zero). The M2 validator must also
  catch *partial* channel dropout — e.g. a subset of real geometry pixels
  reading as background `+inf` / id `0` while the rest of the frame is fine —
  which the per-frame exporter guard is not positioned to see.
- The gameplay debug log (`user://logs/session_*.jsonl`, KB Decision 22) and the
  dataset manifest are **separate artifacts** with separate "session" concepts;
  the recorder must not conflate them.

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
- pinned engine version — **Godot `4.7.2-stable` (`4.7.2.stable.official.ed1daf0bf`), standard build**, recorded in `/.godot-version` (pinned at Milestone 0, 2026-09-03);
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
