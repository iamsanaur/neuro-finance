//! tensor-engine
//!
//! The one place in this workspace that names a concrete ML tensor backend
//! (project spec §5). Every other crate that needs tensors depends on
//! `tensor-engine`, not on `burn` directly, and uses [`Backend`]/[`Device`]
//! rather than a hardcoded `burn::backend::...` type. Swapping backends —
//! the documented next step once there's a model worth benchmarking (see
//! `docs/environment.md`'s "Tensor/ML backend decision") — means changing
//! the two type aliases below and nothing else in the workspace.
//!
//! `burn` itself is re-exported (`tensor_engine::burn`) rather than wrapped
//! API-by-API: hand-wrapping Burn's tensor/module/autodiff surface would be
//! a large amount of code whose only job is forwarding calls, which is the
//! "unnecessary trait abstraction" project spec §4 warns against. The
//! actual isolation this crate provides is at the *dependency* level (one
//! `Cargo.toml` names `burn`, not eighteen) and the *backend-selection*
//! level (one pair of type aliases), which is what §5 asks for.
//!
//! ## Backend decision (recap; full reasoning in `docs/environment.md`)
//!
//! `NdArray` (CPU) by default — deterministic, no driver/Metal variables to
//! debug, and small enough workloads (100 assets, 30-day sequences) that
//! CPU isn't a bottleneck yet. `Autodiff<NdArray>` (not bare `NdArray`) is
//! the default [`Backend`] alias even though nothing trains gradients yet
//! (`training-engine` doesn't exist as of this crate's introduction,
//! Milestone 6) — changing every downstream crate's `Tensor<B, _>` type
//! parameter later, once training-engine needs autodiff, would be a
//! breaking change repeated across the workspace; picking it now costs
//! nothing (forward-only code ignores the autodiff wrapper) and avoids that
//! churn.

pub use burn;

/// The concrete backend without autodiff. Rarely used directly — most code
/// should use [`Backend`], which wraps this in `Autodiff` — but exposed for
/// the rare case (e.g. a benchmark) that specifically wants to measure or
/// exercise the non-differentiable path.
pub type InnerBackend = burn::backend::NdArray<f32>;

/// The backend every other crate should use for `Tensor<Backend, _>`.
pub type Backend = burn::backend::Autodiff<InnerBackend>;

// `Device` is declared on `BackendTypes` (a supertrait of `Backend`), not on
// `Backend` itself — `<T as Backend>::Device` doesn't resolve an
// associated type that's only declared on a supertrait, so this must name
// `BackendTypes` directly.
pub type Device = <InnerBackend as burn::tensor::backend::BackendTypes>::Device;

/// The default (only, for now) device — `NdArray` has no notion of "which
/// GPU," so this always succeeds and is always the same value. Exists so
/// callers don't need to know that.
pub fn device() -> Device {
    Device::default()
}

/// Seeds the backend's RNG (weight initialization, dropout, etc.) for
/// reproducibility (project spec §31: "deterministic seeds"). Must be
/// called before constructing any module that initializes parameters
/// randomly, e.g. before `LinearConfig::init`.
pub fn seed(seed: u64) {
    <InnerBackend as burn::tensor::backend::Backend>::seed(&device(), seed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn device_is_reachable() {
        let _device = device();
    }

    #[test]
    fn seeded_initialization_is_deterministic() {
        seed(7);
        let a: Tensor<InnerBackend, 1> =
            Tensor::random([8], burn::tensor::Distribution::Normal(0.0, 1.0), &device());
        seed(7);
        let b: Tensor<InnerBackend, 1> =
            Tensor::random([8], burn::tensor::Distribution::Normal(0.0, 1.0), &device());
        let diff: f32 = (a - b).abs().sum().into_scalar();
        assert!(
            diff < 1e-9,
            "same seed should give identical random tensors, diff={diff}"
        );
    }
}
