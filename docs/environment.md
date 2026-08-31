# Environment

Recorded 2026-08-31, during initial workspace setup.

## Hardware

| Item | Value |
|---|---|
| Machine | Mac mini |
| OS | macOS 26.2 (Darwin 25.2.0, kernel `25C56`) |
| Architecture | `arm64` (Apple Silicon) |
| CPU | Apple M4, 10 cores |
| RAM | 16 GB |
| GPU | Apple M4 integrated GPU, 10 cores, Metal 4 |
| CUDA | **Not available** (no NVIDIA GPU — Apple Silicon) |
| Boot volume free space | ~13 GB (`/`) |
| Project volume free space | ~648 GB (`/Volumes/Data 1`) — project lives here, not on the boot volume |

## Toolchain

| Item | Value |
|---|---|
| Rust | 1.98.0 (`88d9e12ae`, 2026-08-18), installed via `rustup` to `~/.cargo` |
| Cargo | 1.98.0 |
| clippy | 0.1.98 |
| rustfmt | 1.9.0 |
| C compiler | Apple clang 17.0.0 |
| nvcc | not present (no CUDA toolkit) |

`~/.cargo/bin` is not yet on the default shell PATH — `~/.zshenv` is root-owned and
could not be amended by the installer. Until the user adds it manually
(`export PATH="$HOME/.cargo/bin:$PATH"`), any shell driving this project must
export that PATH explicitly, exactly as this session's build/test/clippy/fmt
commands did.

## Tensor/ML backend decision (§5)

Candidates considered: **Burn**, **Candle**, **tch (libtorch bindings)**, ONNX Runtime.

- **CUDA is not available** on this machine (Apple Silicon), so any CUDA-only
  backend is out. The realistic GPU path here is **Metal**, via `wgpu`.
- **tch** requires a libtorch install (a large external C++ dependency); neither
  libtorch nor a `torch` Python install is present on this machine, and pulling
  one in adds build friction disproportionate to V0.1's needs. Rejected for now.
- **Candle** (HuggingFace) is lightweight and has a Metal backend, but its
  training/autodiff story is secondary to its inference focus.
- **Burn** is a training-first framework: autodiff is a first-class citizen,
  and its `Backend` trait already *is* the small internal interface §5 asks
  for — swapping `NdArray` (CPU) for `Wgpu` (Metal GPU) is a type parameter,
  not a rewrite. It also has an `ndarray`-backed CPU backend with zero exotic
  system dependencies, which keeps V0.1 easy to build anywhere.

**Decision: Burn**, `crates/tensor-engine` wraps it behind a workspace-local
interface (per §5) so the rest of the workspace never imports `burn` directly.

- Default backend for V0.1: `burn-ndarray` (CPU). It's deterministic, has no
  Metal/driver variables to debug, and 100 assets × 30-day sequences is small
  enough that CPU is not a bottleneck yet.
- Secondary backend, to benchmark once the model exists: `burn-wgpu` (Metal).
- This choice is revisited if profiling in later milestones shows CPU is
  actually limiting — see §44 (Performance Engineering).

## Implication for validation

Every "run tests" / "run benchmark" step in this project's workflow must
export `PATH="$HOME/.cargo/bin:$PATH"` (or the shell must be fixed) before
invoking `cargo`.
