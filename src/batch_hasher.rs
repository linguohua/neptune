use rust_gpu_tools::opencl;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::error::{ClError, Error};
use crate::poseidon::SimplePoseidonBatchHasher;
#[cfg(feature = "opencl")]
use crate::proteus::gpu::ClBatchHasher;
#[cfg(feature = "futhark")]
use crate::triton::{cl, gpu::GpuBatchHasher};
use crate::{Arity, BatchHasher, Strength, DEFAULT_STRENGTH};
use bellperson::bls::Fr;
use generic_array::GenericArray;

#[cfg(feature = "futhark")]
use triton::FutharkContext;

pub enum Batcher<A>
where
    A: Arity<Fr>,
{
    Cpu(SimplePoseidonBatchHasher<A>),
    #[cfg(feature = "futhark")]
    OpenCl(GpuBatchHasher<A>),
    #[cfg(feature = "opencl")]
    OpenCl(ClBatchHasher<A>),
}

pub fn mamami() -> GPUSelector {
    use log::{info, error};
    info!("mamami default gpu selector");
    let bus_id = std::env::var("NEPTUNE_DEFAULT_GPU")
        .ok()
        .and_then(|v| match v.parse::<u32>() {
            Ok(bus_id) => Some(bus_id),
            Err(_) => {
                error!("Bus-id '{}' is given in wrong format!", v);
                None
            }
        });
    match bus_id {
        Some(bus_id) => {
            info!(
                "mamami Using device with bus-id {} for creating the GpuSelector...",
                bus_id
            );
            GPUSelector::BusId(bus_id)
        },
        None => GPUSelector::Index(0),
    }
}

impl<A> Batcher<A>
where
    A: Arity<Fr>,
{
    /// Create a new CPU batcher.
    pub fn new_cpu(max_batch_size: usize) -> Self {
        Self::with_strength_cpu(DEFAULT_STRENGTH, max_batch_size)
    }

    /// Create a new CPU batcher with a specified strength.
    pub fn with_strength_cpu(strength: Strength, max_batch_size: usize) -> Self {
        Self::Cpu(SimplePoseidonBatchHasher::<A>::new_with_strength(
            strength,
            max_batch_size,
        ))
    }

    /// Create a new GPU batcher for an arbitrarily picked device.
    #[cfg(feature = "futhark")]
    pub fn pick_gpu(max_batch_size: usize) -> Result<Self, Error> {
        let futhark_context = cl::default_futhark_context()?;
        Ok(Self::OpenCl(GpuBatchHasher::<A>::new_with_strength(
            futhark_context,
            DEFAULT_STRENGTH,
            max_batch_size,
        )?))
    }

    /// Create a new GPU batcher for an arbitrarily picked device.
    #[cfg(feature = "opencl")]
    pub fn pick_gpu(max_batch_size: usize) -> Result<Self, Error> {
        let all = opencl::Device::all();
        let device = all.first().ok_or(Error::ClError(ClError::DeviceNotFound))?;
        Self::new(device, max_batch_size)
    }

    #[cfg(feature = "futhark")]
    /// Create a new GPU batcher for a certain device.
    pub fn new(device: &opencl::Device, max_batch_size: usize) -> Result<Self, Error> {
        Self::with_strength(device, DEFAULT_STRENGTH, max_batch_size)
    }

    #[cfg(feature = "opencl")]
    /// Create a new GPU batcher for a certain device.
    pub fn new(device: &opencl::Device, max_batch_size: usize) -> Result<Self, Error> {
        Self::with_strength(device, DEFAULT_STRENGTH, max_batch_size)
    }

    #[cfg(feature = "futhark")]
    /// Create a new GPU batcher for a certain device with a specified strength.
    pub fn with_strength(
        device: &opencl::Device,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        let futhark_context = cl::futhark_context(&device)?;
        Ok(Self::OpenCl(GpuBatchHasher::<A>::new_with_strength(
            futhark_context,
            strength,
            max_batch_size,
        )?))
    }

    #[cfg(feature = "opencl")]
    /// Create a new GPU batcher for a certain device with a specified strength.
    pub fn with_strength(
        device: &opencl::Device,
        strength: Strength,
        max_batch_size: usize,
    ) -> Result<Self, Error> {
        Ok(Self::OpenCl(ClBatchHasher::<A>::new_with_strength(
            &device,
            strength,
            max_batch_size,
        )?))
    }
}

impl<A> BatchHasher<A> for Batcher<A>
where
    A: Arity<Fr>,
{
    fn hash(&mut self, preimages: &[GenericArray<Fr, A>]) -> Result<Vec<Fr>, Error> {
        match self {
            Batcher::Cpu(batcher) => batcher.hash(preimages),
            #[cfg(any(feature = "futhark", feature = "opencl"))]
            Batcher::OpenCl(batcher) => batcher.hash(preimages),
        }
    }

    fn max_batch_size(&self) -> usize {
        match self {
            Batcher::Cpu(batcher) => batcher.max_batch_size(),
            #[cfg(any(feature = "futhark", feature = "opencl"))]
            Batcher::OpenCl(batcher) => batcher.max_batch_size(),
        }
    }
}
