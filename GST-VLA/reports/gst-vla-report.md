# RGB-Only Embodied 3D Reasoning with Uncertainty-Aware 3D Gaussian Tokens and Graph-Grounded VLM Planning

## Executive Summary

This report proposes a novel ACCV-suitable methodology for **embodied 3D reasoning and action from RGB-only input** by turning monocular perception into a **calibrated, uncertainty-aware 3D token space** that can be consumed by a **vision-language reasoning core** and executed by a **diffusion-based action controller with safety shielding**. The central idea is to bridge the “RGB-only → reliable 3D → safe long-horizon action” gap by (i) pairing a strong semantic encoder (open-vocabulary, mask-capable) with a modern monocular depth foundation model, (ii) converting their outputs into **3D Gaussian Spatial Tokens** (geometry + semantics + uncertainty), (iii) maintaining a **spatial knowledge graph** grounded in those Gaussians to reduce hallucinations and enable explicit constraint reasoning, and (iv) using **receding-horizon diffusion policy control** with **trajectory verification** and optional barrier-function shielding for safety. citeturn0search7turn0search6turn0search8turn2search0turn10search24

The proposal is designed to be incremental enough to be implementable with current tooling, yet novel enough for an ACCV paper through: **(a)** a new representation layer (Gaussian Spatial Tokens + graph), **(b)** fusion and learning objectives tailored to RGB-only uncertainty, **(c)** an efficiency plan (token pruning + low-bit quantization for the reasoning core), and **(d)** a sim-to-real training/evaluation stack centered on egocentric video and robotics datasets. citeturn5search0turn5search5turn4search4turn3search10turn2search11

Unspecified elements (to be filled for submission): robot platform (arm/mobile manipulator), camera intrinsics/extrinsics source, action space (joint/EE pose), diffusion horizon length, token dimensionalities, and compute budget.

## Problem statement and research gaps vs state of the art

The goal is: **Given only RGB observations**, produce **3D-grounded, long-horizon, safe robot behavior** that supports (i) spatial reasoning (occlusion, collision, support relations), (ii) language-conditioned planning, and (iii) closed-loop execution on hardware.

Modern embodied VLM systems show that web-scale vision-language pretraining can improve robotic generalization and semantic reasoning (e.g., embodied multimodal language models and VLA models). However, these systems are often weak in **explicit 3D grounding** (e.g., they may not maintain a metric-consistent world model), making collision-aware or occlusion-aware reasoning brittle when only RGB is available. citeturn1search0turn1search1turn1search2

At the same time, several successful “LLM/VLM + 3D structure” approaches rely on **depth sensors** or explicit 3D inputs (RGB-D, point clouds, voxel maps) to build grounded value maps or scene graphs. For example, trajectory-synthesis approaches grounded in 3D value maps assume access to reliable 3D perception, and planning with 3D scene graphs assumes a stable geometric substrate. citeturn2search2turn7search4turn7search0

For **RGB-only**, the key gaps are:

**Geometry gap (metric and uncertainty):** Monocular depth estimation has advanced substantially with foundation models (e.g., Depth Anything V2), but RGB-only geometry remains uncertain, scene-dependent, and vulnerable to scale ambiguity. Depth Anything V2 explicitly emphasizes generalization/robustness and provides efficient models across scales, including metric-depth fine-tuning variants, but the downstream robotics pipeline must still manage uncertainty and drift. citeturn0search0turn0search8turn0search4

**Embodied generalization gap:** Simulation benchmarks highlight that depth-like sensing can be critical for cross-dataset generalization in navigation, underscoring why RGB-only agents often struggle without additional geometric priors. citeturn3search7turn3search3

**Representation gap for VLM reasoning:** VLM cores typically consume 2D tokens; injecting 3D structure is nontrivial. Recent work on 3D scene understanding with large models increasingly uses point clouds/3D tokens directly (3D-VLMs), but robotics needs a representation that is (i) updatable online, (ii) efficient, and (iii) compatible with language reasoning and control. citeturn7search10turn7search2turn5search2

**Control gap (long-horizon + safety):** Diffusion-based policies are compelling for multimodal action generation and stability, and extensions incorporate 3D representations (e.g., DP3). But for RGB-only, the control stack must be paired with an explicit safety/verification layer because perception errors can cause catastrophic collisions. citeturn2search0turn2search1turn10search11turn10search24

These gaps motivate an architecture that treats monocular geometry as a **probabilistic 3D substrate** (not truth), and uses that substrate to **ground** both reasoning and safety checks.

## Novel contributions for an ACCV submission

The following contributions are phrased as ACCV-style claims; each is intended to be measurable via ablations (proposed later):

**Architectural contributions**
- **Gaussian Spatial Tokenizer (GST):** A new mid-level representation that converts RGB-only semantics + pseudo-depth into **3D Gaussian Spatial Tokens** carrying mean position, covariance/uncertainty, appearance, and semantic embeddings. This is inspired by the efficiency and expressive power of 3D Gaussian representations (including their adoption in SLAM and robotics), but re-targeted to **token-level grounding for VLM+policy pipelines**. citeturn5search0turn5search5turn5search9  
- **Graph-grounded reasoning interface:** A **spatial knowledge graph** built from Gaussian tokens (objects/surfaces/free-space nodes, relation edges) that (i) provides a compact symbolic handle for the VLM planner, and (ii) supports explicit constraint checks (collision, support, containment) to reduce 3D hallucination. This directly aligns with the demonstrated usefulness of 3D scene graphs for scalable planning, while adapting it to RGB-only uncertainty. citeturn7search4turn7search0

**Learning objective contributions**
- **Uncertainty-calibrated geometry learning:** Train the GST to produce **calibrated covariances** via heteroscedastic likelihood losses and multi-view/temporal consistency on large egocentric video (where available), leveraging pseudo-depth supervision from a monocular depth foundation model rather than requiring real depth sensors at inference. citeturn0search8turn3search10turn4search4turn4search0  
- **Spatial-language alignment at 3D token level:** Contrastive / matching losses align text-referred entities with Gaussian clusters, building on the success of language-image pretraining losses (e.g., CLIP/SigLIP) but relocating the alignment target from global image embeddings to **3D grounded entities**. citeturn0search7turn0search6turn7search4

**Fusion strategy contributions**
- **Two-stage fusion with token economy:** (i) local fusion to create per-object/per-surface Gaussian tokens (compressed from dense pixels), then (ii) global fusion where the VLM attends over a **small set of spatial-semantic tokens**, optionally using adaptive tokenization ideas to keep inference cost bounded. citeturn6search6turn5search2

**Efficiency and deployment contributions**
- **Low-bit reasoning-core deployment plan:** Apply modern post-training quantization / 4-bit finetuning methods (e.g., GPTQ / AWQ / QLoRA styles) to the VLM reasoning core and possibly the fusion transformer, enabling on-robot inference while maintaining plan quality. citeturn8search2turn8search1turn8search0

**Safety and constraint-check contributions**
- **Verification-first action execution:** A trajectory checker that uses the Gaussian map (and its uncertainty) to implement conservative collision margins; optionally integrate barrier-function or MPC-style shielding as a drop-in “safety filter.” citeturn10search11turn10search24turn10search17

**Sim-to-real contributions**
- **RGB-only sim-to-real recipe:** Domain randomization for visual diversity plus data aggregation (DAgger-style corrections) for compounding-error reduction in long-horizon tasks, explicitly tailored to RGB-only geometry errors. citeturn9search0turn9search1

## Pipeline diagram and component-level design

The diagram below formalizes your stated diagram into an implementable pipeline, adding two missing pieces that are critical for RGB-only 3D reasoning: **pose estimation / world alignment** and **uncertainty-aware mapping**. Pose can be obtained with classical or learned monocular SLAM/VO (e.g., ORB-SLAM3, DROID-SLAM) depending on your engineering preference. citeturn9search7turn9search2

```mermaid
flowchart LR
  I[RGB frame I_t] --> S[Semantic encoder\n(CLIP/SigLIP/DINOv2 features)]
  I --> M[Mask generator\n(SAM / Mask2Former optional)]
  I --> D[Pseudo-depth expert\n(Depth Anything V2)]
  I --> P[Monocular pose/VO\n(ORB-SLAM3 / DROID-SLAM optional)]

  S --> T2D[2D semantic tokens\n(patch/object tokens)]
  M --> Mask[Instance/region masks]
  D --> Z[Depth + (optional) normals + uncertainty]
  P --> Twc[Camera pose T_WC(t)]

  T2D --> GST[Gaussian Spatial Tokenizer\n(unproject + fuse)]
  Mask --> GST
  Z --> GST
  Twc --> GST

  GST --> GMap[3D Gaussian map\n(sliding window / persistent)]
  GMap --> SKG[Spatial knowledge graph\n(objects, surfaces, relations)]

  T2D --> Fuse[Spatial-semantic fusion\n(cross-attn over 2D + 3D tokens)]
  GMap --> Fuse
  SKG --> Fuse

  Fuse --> VLM[VLM reasoning core\n(3D grounded planning)]
  VLM --> Plan[High-level plan\n(subgoals + constraints)]

  Plan --> DP[Diffusion policy head\n(receding horizon)]
  Fuse --> DP
  DP --> Check[Trajectory check + safety shield]
  Check --> Act[Robot hardware execution]
  Act --> FB[State feedback]
  FB --> DP
  FB --> P
```

This pipeline explicitly separates **representation learning** (GST + map + graph) from **decision modules** (VLM planner + diffusion controller), enabling clean ablations and reducing confounds typical in end-to-end robot learning. citeturn2search0turn5search5turn7search4

### Component-level specification

**Semantic path (RGB → semantic tokens)**  
Use a vision transformer backbone pretrained with language supervision or self-supervision (e.g., CLIP/SigLIP-style dual encoders, or DINOv2 features). Vision Transformers operate by converting an image into patch tokens, which is convenient for downstream token fusion. citeturn6search0turn0search7turn0search6turn0search1  
For semantic masks, leverage promptable segmentation or universal segmentation to create object/region proposals; this is particularly important to keep the later 3D token count manageable. citeturn11search0turn11search3

**Pseudo-depth path (RGB → depth priors)**  
Use Depth Anything V2 to produce dense depth estimates (and optionally derived normals); the model’s focus on robust monocular depth and reported efficiency makes it a plausible “expert” in an RGB-only robotics stack. citeturn0search0turn0search8turn0search4  
To incorporate egocentric geometry priors beyond depth (your “Ego3D norms” concept), you can optionally train/finetune a normals/gravity-direction head on egocentric datasets such as EDINA (depth+normals+gravity), which directly targets the tilted viewpoints and dynamic hands common in wearable/robot-mounted perspectives. citeturn4search4turn4search7

**Pose / alignment (RGB → T\_WC(t))**  
For persistent 3D reasoning, you need to place observations in a common frame. Monocular SLAM/VO options include ORB-SLAM3 (feature-based) and DROID-SLAM (deep SLAM with dense bundle adjustment), both supporting monocular settings. citeturn9search7turn9search2turn9search6

**Coordinate frames and transforms**  
Define at minimum:
- **Camera frame** \(C_t\) at time \(t\)  
- **Robot base frame** \(B\) (fixed on robot)  
- **World/odometry frame** \(W\)

Assume known extrinsics \(T_{BC}\) from calibration and estimated \(T_{WC_t}\) from VO/SLAM (or from robot state if available). For a pixel \(u=(x,y)\) with depth \(z\) and intrinsics \(K\):
\[
p_{C_t} = z \, K^{-1} [x,y,1]^T,\quad p_W = T_{WC_t}\, p_{C_t}.
\]
This gives world-aligned 3D points that become means of Gaussian tokens.

**Uncertainty modeling and 3D Gaussian Spatial Tokens**  
A single Gaussian Spatial Token \(g_i\) is:
- mean position \(\mu_i \in \mathbb{R}^3\) (world frame)  
- covariance \(\Sigma_i \in \mathbb{R}^{3\times3}\) (anisotropic; encodes depth/pose uncertainty)  
- semantic embedding \(s_i \in \mathbb{R}^{d_s}\) (from semantic encoder)  
- appearance \(c_i \in \mathbb{R}^3\) (optional, RGB)  
- weight/opacity \(w_i \in [0,1]\) (confidence / map contribution)

This is conceptually aligned with 3D Gaussian representations used for real-time rendering and increasingly for mapping/SLAM, but here it is explicitly cast as a **token interface** for reasoning and control. citeturn5search0turn5search5turn5search8

A practical covariance construction:
- start from predicted depth variance \(\sigma_z^2\) (learned head or ensemble)  
- propagate through unprojection Jacobian to get \(\Sigma_i\) in the camera frame  
- add pose covariance \(\Sigma_{pose}\) from VO/SLAM  
- transform to world frame

Training can calibrate \(\sigma_z\) with heteroscedastic likelihood losses; Depth Anything V2 provides a strong base depth signal, but your novelty is **calibrated downstream uncertainty** for safe action. citeturn0search8turn10search24

**Spatial knowledge graph (SKG)**  
Build a graph \(G=(V,E)\) where nodes represent:
- object instances (clusters of Gaussians + semantic label distribution)  
- planar/support surfaces (table, floor)  
- free-space and obstacles  
Edges encode relations (above/on/inside/near, connectivity, collision risk). This mirrors the utility of 3D scene graphs for scalable planning, but your SKG is grounded in Gaussian tokens and updated online. citeturn7search4turn7search0turn5search5

**Spatial-semantic fusion**  
Use cross-attention between:
- 2D semantic tokens (patch/object tokens)  
- 3D Gaussian tokens (map-local subset)  
- SKG tokens (node/edge embeddings)

To control compute, compress dense tokens into a small set of learned tokens (e.g., TokenLearner-like selection) or mask-guided pooling before global reasoning. citeturn6search6turn11search0

**VLM reasoning core**  
Adopt a multimodal instruction-tuned VLM family (LLaVA-style architectures are a canonical reference for “vision encoder + LLM” fusion), but **constrain its output space**: require structured subgoals in \(W\) and explicit constraints tied to SKG entities to prevent free-form hallucinated plans. citeturn7search3turn7search15turn7search4

**Action controller and verification**  
Use a diffusion policy head (Diffusion Policy) or a 3D-aware diffusion policy variant (DP3) conditioned on fused tokens and robot state, executed in receding horizon. citeturn2search0turn2search1  
Then apply trajectory checking: collision and constraint validation against the Gaussian map (with uncertainty-inflated safety margins), plus optional integration with trajectory optimization or barrier-function style safety mechanisms. citeturn10search11turn10search24turn10search7

**Efficiency and quantization detail**  
Quantize the reasoning core (and possibly fusion layers) using modern low-bit methods:
- GPTQ-style post-training quantization for inference citeturn8search2  
- AWQ-style activation-aware quantization citeturn8search1turn8search5  
- QLoRA-style 4-bit finetuning to retain performance while enabling small-GPU training citeturn8search0turn8search8

## Training regime, datasets, objectives, and metrics

The training strategy is deliberately **staged** to (i) exploit large-scale pretrained models, (ii) reduce robot-data needs, and (iii) make RGB-only 3D grounding learnable.

```mermaid
flowchart TD
  A[Stage A: Initialize frozen experts\nSemantic encoder + Depth Anything V2] --> B
  B[Stage B: Train GST + fusion on video\n(temporal + multi-view consistency)] --> C
  C[Stage C: Synthetic pretraining in sim\n(3D supervision: occupancy, instances, relations)] --> D
  D[Stage D: Plan grounding training\n(SKGraph-conditioned plan supervision)] --> E
  E[Stage E: Policy learning\n(imitation + diffusion) + safety checker tuning] --> F
  F[Evaluation\n(sim benchmarks + real robot)] --> G
  G[Failure analysis + DAgger-style data aggregation\n(target hard states)] --> E
```

This aligns with known robotics practice: large-scale pretraining + task finetuning + closed-loop data aggregation to reduce compounding errors. citeturn9search1turn2search0turn1search3

### Datasets and how they fit the pipeline

A compact “dataset plan” suitable for an ACCV methodology section is summarized below.

| Dataset / Source | Modality | Why it matters in this pipeline | Suggested usage stage |
|---|---|---|---|
| Ego4D | egocentric RGB video + narrations/benchmarks | Large-scale egocentric visual diversity; supports temporal consistency pretraining and language grounding. citeturn3search10turn3search2turn3search14 | Stage B / D |
| Ego-Exo4D | ego + exo multi-view video + language | Multi-view signals are valuable for 3D-aware representation learning and correspondence. citeturn4search6turn4search3 | Stage B |
| EDINA (EgoDepthNormal) | egocentric RGBD + normals + gravity | Direct supervision for surface normals/gravity priors under ego tilt and hand occlusions. citeturn4search4turn4search7 | Stage B |
| Habitat | photorealistic 3D simulation platform | Efficient embodied simulation; supports navigation/instruction tasks; highlights depth importance for generalization (useful for motivation + baselines). citeturn3search3turn3search7 | Stage C / F |
| YCB object set | physical objects + models/protocols | Standard manipulation benchmarking objects and protocols; supports sim object realism and evaluation. citeturn3search4turn3search8 | Stage C / F |
| Open X-Embodiment | multi-robot real trajectories | Large-scale robot demonstrations; useful for policy pretraining/finetuning and benchmarking generalist policies. citeturn2search11turn2search3 | Stage E |
| Real robot data (your lab) | RGB + proprioception + actions | Necessary for calibration, domain gap closure, and final results; scope and robot are unspecified. | Stage E / F |

### Loss functions and training objectives

**Stage B: GST + fusion pretraining (RGB-only video, weak 3D)**  
Use a mixture of objectives:
- **Temporal consistency for geometry:** enforce consistency of Gaussian means across adjacent frames after warping with estimated poses (VO/SLAM). citeturn9search2turn9search7  
- **Photometric / feature reprojection:** render a view from the Gaussian map and match observed RGB/features (inspired by Gaussian rendering’s differentiability and efficiency). citeturn5search0turn5search5  
- **Uncertainty calibration:** heteroscedastic regression loss (negative log-likelihood) for depth/3D errors to learn meaningful \(\Sigma\). citeturn10search24  
- **Language grounding at entity level:** contrastive matching between text entities and Gaussian clusters, seeded by CLIP/SigLIP-style alignments. citeturn0search7turn0search6

**Stage C: Synthetic 3D supervision (sim)**  
In simulation, supervise:
- occupancy/free-space labels and collision risk  
- instance-driven clustering quality (object identity stability)  
- relation classification for SKG edges

Habitat enables high-throughput photorealistic simulation and task definition, which makes it a reasonable backbone for Stage C. citeturn3search3turn3search7

**Stage D: Plan grounding**  
Train the planner to output structured subgoals constrained by the SKG (e.g., “go to node X, put object node Y on surface node Z”), drawing inspiration from systems that ground LLM planning in 3D scene graphs. citeturn7search4turn7search0

**Stage E: Diffusion policy learning + safety checker tuning**  
- imitation learning for diffusion policy (Diffusion Policy / DP3-like conditioning) citeturn2search0turn2search1  
- receding-horizon execution and action-sequence prediction losses (standard in diffusion policy formulations) citeturn2search0  
- “shield loss”: penalize predicted trajectories that violate collision margins derived from Gaussian map uncertainty; optionally incorporate trajectory optimization priors (CHOMP/STOMP-style smoothness + obstacle costs) to stabilize execution. citeturn10search11turn10search7

### Metrics aligned to your claims

Use metrics that separate perception, reasoning, and control:
- **3D grounding quality:** relative depth error / scale consistency (where available), temporal stability of mapped objects, uncertainty calibration (e.g., NLL, calibration error). citeturn0search8turn4search4  
- **Navigation:** success rate, SPL-style efficiency scores in Habitat-like tasks. citeturn3search7  
- **Manipulation:** success rate on YCB-style protocols, collision rate, time-to-completion. citeturn3search4  
- **Long-horizon planning:** task completion under occlusions and multi-step constraints; plan validity rate (constraint satisfaction). citeturn7search4turn2search2  
- **Compute:** end-to-end latency and memory; quantify impact of quantization (4-bit/8-bit). citeturn8search0turn8search2turn8search3

## Experiments, baselines, ablations, and evaluation protocol

### Baseline families

Design baselines that isolate *why* your method works:

**RGB-only VLA / generalist policy baselines**  
Compare against VLA-style systems that map RGB(+language) directly to actions (conceptually aligned with RT-2-like pipelines) and against generalist diffusion-transformer policies pretrained on large robot datasets (Octo / Open X-Embodiment). citeturn1search1turn1search3turn2search11

**Explicit 3D planning baselines (with depth) adapted to RGB-only**  
Include systems that use 3D value maps or 3D scene graphs, but replace real depth with monocular pseudo-depth, to quantify the “RGB-only penalty” and whether your uncertainty-aware Gaussian substrate closes it. citeturn2search2turn7search4

**3D diffusion baselines**  
DP3 is a natural baseline for “3D representation + diffusion policy,” though it typically assumes actual 3D inputs; you can test a pseudo-depth point-cloud variant vs your Gaussian tokens. citeturn2search1turn2search5

### Key ablations mapped to contributions

Ablations should be cut along the axes you claim are novel:

| Ablation | What is removed/changed | What it tests | Primary success signals |
|---|---|---|---|
| No-Gaussian (points only) | Replace Gaussian tokens with point tokens | value of covariance/uncertainty-bearing representation | collision rate ↓, occlusion success ↑ |
| No-uncertainty | Fix \(\Sigma\) or ignore it in planning/checking | whether calibrated uncertainty is essential | safety violations ↓ with uncertainty |
| No-SKG | Remove graph; VLM attends only to tokens | graph’s role in constraint reasoning | hallucination/invalid-plan rate ↓ |
| Fusion variant | early fusion vs late fusion vs two-stage | which fusion strategy is best | task success vs latency |
| Depth expert swap | different monocular depth models / finetuning | dependence on depth quality | robustness across domains |
| Quantization | fp16 vs int8 vs 4-bit | deployability vs accuracy tradeoff | latency/memory vs success |
| Safety checker off | disable trajectory verification | necessity of verification | collision rate ↑ without checker |

Several of these comparisons are motivated by existing evidence that (i) 3D structure helps language planning (3D scene graphs/value maps), (ii) diffusion policies benefit from 3D representations, and (iii) low-bit quantization is feasible for transformer inference. citeturn7search4turn2search1turn8search2turn8search0

### Tasks and protocols

Cover a spread that ACCV reviewers will recognize as meaningful:

**Object manipulation:** pick/place, container insertion, tool use, cluttered scenes, evaluated with YCB objects and standardized protocols where possible. citeturn3search4

**Navigation + object search:** point-goal and language-referred object navigation in Habitat-like environments, emphasizing occlusion and viewpoint change. citeturn3search3turn3search7

**Occlusion handling:** “retrieve occluded object,” “place behind obstacle,” evaluate with partial observability; report both success and safety.

**Long-horizon multi-step instructions:** tasks requiring subgoal chaining and constraint satisfaction (e.g., “pick mug, avoid spilling area, place on shelf”), inspired by the motivation in grounded planning works. citeturn1search2turn7search4

**Evaluation protocol essentials**
- fixed random seeds and scene splits (train/val/test)  
- cross-domain evaluation (textures/lighting/cameras)  
- report both task success and safety (collisions, constraint violations)

Compute budget should be reported in ACCV style (GPUs, days). Your exact budget is unspecified; the staged approach is intentionally flexible, and quantization-aware finetuning methods can reduce GPU memory requirements. citeturn8search0turn8search2

## Expected results, limitations, failure modes, and mitigation

### Expected results that follow from the design

If the Gaussian tokens and SKG actually ground reasoning, you should see:
- improved **occlusion robustness** and fewer “phantom-object” actions (because planning queries the map/graph rather than hallucinating) citeturn7search4turn5search5  
- reduced **collision rate** under long-horizon tasks (trajectory checker “shields” errors from VLM/policy) citeturn10search11turn10search24  
- better **sim-to-real transfer** versus RGB-only end-to-end policies, due to domain randomization + geometry priors + data aggregation. citeturn9search0turn9search1

### Known limitations in RGB-only 3D pipelines

**Scale ambiguity and drift:** monocular depth and monocular VO can drift; Depth Anything V2 provides strong depth priors, but the pipeline must still manage scale and pose uncertainty. citeturn0search8turn9search7turn9search2

**Failure in reflective/textureless regions:** monocular depth commonly struggles in low-texture, specular, or thin-structure scenes; the mitigation is to reflect that uncertainty into \(\Sigma\) and enlarge safety margins rather than trusting geometry. citeturn0search8turn10search24

**Dynamic objects and hands:** egocentric/robot-mounted views often include hands/arms and fast motion; egocentric depth/normal datasets explicitly address these issues and can be used to improve robustness. citeturn4search4turn4search1

**Latency/compute:** VLM reasoning plus mapping can be expensive; token compression and quantization are necessary for real-time robotics. citeturn6search6turn8search2turn8search3

### Mitigation strategies to include in the paper

**Uncertainty-aware conservative planning:** inflate obstacles by uncertainty; reject actions that enter high-uncertainty regions unless the task requires exploratory behavior.

**Fallback behaviors:** when uncertainty spikes, fall back to safer primitive skills (slow approach, re-observe, change viewpoint), consistent with grounded-skill ideas in embodied planning systems. citeturn1search2

**Active viewpoint selection:** use information gathering (move camera to reduce ambiguity) as a meta-action; this is especially important in RGB-only settings. (If you include this, formalize it as a small add-on module; otherwise keep it out of scope.)

**Safety shields / barrier functions:** if you need stronger safety guarantees, integrate control barrier functions or MPC-style filters as an optional layer. citeturn10search24turn10search17

### Suggested figures and tables with captions and brief drafting text

Below are drafting-ready figure/table plans that match ACCV expectations and your requested list.

**Figure: System architecture overview**  
*Caption:* “Overview of the RGB-only embodied 3D reasoning system. RGB observations are processed by a semantic encoder and a monocular depth expert; outputs are fused into uncertainty-aware 3D Gaussian Spatial Tokens and a spatial knowledge graph. A VLM planner produces grounded subgoals and constraints, executed via a diffusion policy with trajectory verification.” citeturn0search8turn5search5turn2search0turn7search4  
*Brief text:* Emphasize modularity: swapping encoders, ablatable map/graph, and explicit safety checks.

**Table: Token flow and representation schema**  
*Caption:* “Token types, coordinate frames, and update rates used throughout the pipeline.”  
*Brief text:* Declare a small number of token types to keep the method crisp.

Example schema (fill dims later; hyperparameters unspecified):

| Token | Symbol | Frame | Core fields |
|---|---|---|---|
| 2D semantic token | \(t^{2D}_i\) | image | patch embedding, mask id, confidence |
| Gaussian spatial token | \(g_i\) | world | \(\mu_i,\Sigma_i,s_i,c_i,w_i\) |
| SKG node token | \(v_k\) | world | centroid, extents, class distribution, affordances |
| Plan token | \(p_j\) | world | subgoal pose, constraint refs (node ids) |
| Action sequence | \(a_{1:H}\) | robot | EE pose/joints over horizon |

**Table: Dataset and supervision sources**  
*Caption:* “Training data sources and which module they supervise (representation, planning, control).” citeturn3search10turn4search6turn3search3turn3search4turn2search11  
*Brief text:* Highlight that inference remains RGB-only even if some training uses depth in sim or egocentric RGBD.

**Table: Ablation matrix**  
*Caption:* “Ablations isolating the effect of Gaussian tokens, uncertainty calibration, SKG grounding, and safety verification.”  
*Brief text:* Present 6–8 ablations max; tie each directly to a claim.

**Figure: Training and evaluation flowchart**  
*Caption:* “Staged training: frozen experts → self-supervised RGB-only 3D token learning → sim 3D supervision → grounded planning → diffusion policy + safety tuning → sim and real evaluation with data aggregation.” citeturn9search1turn2search0turn3search3  
*Brief text:* Stress that staged training reduces robot-data needs.

**Table: Project timeline**  
*Caption:* “Implementation timeline for reproducing results: representation learning, planning integration, policy learning, sim benchmarks, real-robot validation.”  
*Brief text:* Keep it high-level; durations/hardware unspecified.

### Prioritized sources for ACCV/ICCV/CVPR/NeurIPS-style positioning

The following references map cleanly to the narrative and should be cited early in the paper:

Foundational vision-language and vision backbones: CLIP (contrastive pretraining from image-text) and SigLIP (sigmoid loss scaling), plus DINOv2 for robust self-supervised ViT features. citeturn0search7turn0search6turn0search1

Monocular depth: Depth Anything V2 (robust monocular depth foundation model; efficiency-oriented and scalable), plus egocentric depth/normal priors via EDINA. citeturn0search8turn0search0turn4search4

Embodied VLM / VLA: PaLM-E and RT-2 as canonical exemplars of grounding language in embodied tasks and mapping RGB(+language) to actions. citeturn1search0turn1search1

Generalist robot policies and robot datasets: Open X-Embodiment and Octo (large diffusion-transformer policies). citeturn2search11turn1search3turn1search7

3D grounded planning: VoxPoser (3D value maps + language constraints) and SayPlan (LLM planning grounded in 3D scene graphs). citeturn2search2turn7search4turn7search0

3D Gaussian representations for mapping: 3D Gaussian Splatting; GS-SLAM; and robotics-oriented Gaussian manipulation/mapping examples (use as related work to justify Gaussian tokens). citeturn5search0turn5search5turn5search9

Diffusion policies for control: Diffusion Policy and 3D Diffusion Policy (DP3) as direct lineage for the “diffusion policy head.” citeturn2search0turn2search1

Safety/verification and sim-to-real: CHOMP/STOMP-style motion planning references for trajectory verification plus domain randomization + DAgger for sim-to-real robustness. citeturn10search11turn10search7turn9search0turn9search1

For venue framing, position the submission as a systems-and-methods contribution aligned with the entity["organization","Asian Conference on Computer Vision","computer vision conference"] and related top venues such as the entity["organization","IEEE/CVF International Conference on Computer Vision","computer vision conference"], the entity["organization","IEEE/CVF Conference on Computer Vision and Pattern Recognition","computer vision conference"], and the entity["organization","Conference on Neural Information Processing Systems","machine learning conference"], while grounding embodied robotics claims in the entity["organization","Conference on Robot Learning","robot learning conference"] and robotics mapping/control literature. citeturn3search10turn1search3turn5search5turn2search0