use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::error::{ClError, Error};
use crate::poseidon::SimplePoseidonBatchHasher;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::proteus::gpu::ClBatchHasher;
use crate::{Arity, BatchHasher, NeptuneField, Strength, DEFAULT_STRENGTH};
use ec_gpu_gen::rust_gpu_tools::Device;
use ff::PrimeField;
use generic_array::GenericArray;

pub enum Batcher<F, A>
where
    F: NeptuneField,
    A: Arity<F>,
{
    Cpu(SimplePoseidonBatchHasher<F, A>),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    OpenCl(ClBatchHasher<F, A>),
}

pub fn mamami(user_gpu_id: &str) -> Result<&'static Device, Error> {
    use log::{info, error};
    use std::convert::TryFrom;
    use ec_gpu_gen::rust_gpu_tools::UniqueId;

    let id_str = if user_gpu_id.len() > 0 {
        user_gpu_id.to_string()
    } else {
        std::env::var("NEPTUNE_DEFAULT_GPU")
        .unwrap_or_default()
    };

    info!("mamami gpu selector, unique id:{}", id_str);

    if id_str.len() > 0 {
        let id = UniqueId::try_from(id_str.as_str())?;
        if let Some(d) = Device::by_unique_id(id) {
            return Ok(d);
        }
    }

    let dd = Device::all();
    if dd.len() > 0 {
        return Ok(dd[0]);
    }

    let msg = format!("mamami gpu selector, no device found for:{}", id_str);
    return Err(Error::Other(msg));
}

impl<F, A> Batcher<F, A>
where
    F: NeptuneField,
    A: Arity<F>,
{
    /// Create a new CPU batcher.
    pub fn new_cpu(max_batch_size: usize) -> Self {
        Self::with_strength_cpu(DEFAULT_STRENGTH, max_batch_size)
    }

    /// Create a new CPU batcher with a specified strength.
    pub fn with_strength_cpu(strength: Strength, max_batch_size: usize) -> Self {
        Self::Cpu(SimplePoseidonBatchHasher::<F, A>::new_with_strength(
            strength,
            max_batch_size,
        ))
    }

    /// Create a new GPU batcher for an arbitrarily picked device.
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    pub fn pick_gpu2(gpu_id: &str, max_batch_size: usize) -> Result<Self, Error> {
        //let all = opencl::Device::all();
        //let device = all.first().ok_or(Error::ClError(ClError::DeviceNotFound))?;
        let device = mamami(gpu_id)?;
        Self::new(device, max_batch_size)
    }

    /// Create a new GPU batcher for an arbitrarily picked device.
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    pub fn pick_gpu(max_batch_size: usize) -> Result<Self, Error> {
        let all = Device::all();
        let device = all.first().ok_or(Error::ClError(ClError::DeviceNotFound))?;
        Self::new(device, max_batch_size)
    }

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    /// Create a new GPU batcher for a certain device.
    pub fn new(device: &Device, max_batch_size: usize) -> Result<Self, Error> {
        Self::with_strength(device, DEFAULT_STRENGTH, max_batch_size)
    }

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    /// Create a new GPU batcher for a certain device with a specified strength.
    pub fn with_strength(
        device: &Device,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        Ok(Self::OpenCl(ClBatchHasher::<F, A>::new_with_strength(
            device,
            strength,
            max_batch_size,
        )?))
    }
}

impl<F, A> BatchHasher<F, A> for Batcher<F, A>
where
    F: NeptuneField,
    A: Arity<F>,
{
    fn hash(&mut self, preimages: &[GenericArray<F, A>]) -> Result<Vec<F>, Error> {
        match self {
            Batcher::Cpu(batcher) => batcher.hash(preimages),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            Batcher::OpenCl(batcher) => batcher.hash(preimages),
        }
    }

    fn max_batch_size(&self) -> usize {
        match self {
            Batcher::Cpu(batcher) => batcher.max_batch_size(),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            Batcher::OpenCl(batcher) => batcher.max_batch_size(),
        }
    }
}
