use crate::batch_hasher::{Batcher, BatcherType};
use crate::error::Error;
use crate::poseidon::{Poseidon, PoseidonConstants};
use crate::tree_builder::{TreeBuilder, TreeBuilderTrait};
use crate::{Arity, BatchHasher};
use bellperson::bls::{Bls12, Fr};
use ff::Field;
use generic_array::GenericArray;
use log::info;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use rust_gpu_tools::opencl::GPUSelector;

pub trait ColumnTreeBuilderTrait<ColumnArity, TreeArity>
where
    ColumnArity: Arity<Fr>,
    TreeArity: Arity<Fr>,
{
    fn add_columns(&mut self, columns: &[GenericArray<Fr, ColumnArity>]) -> Result<(), Error>;
    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<Fr, ColumnArity>],
    ) -> Result<(Vec<Fr>, Vec<Fr>), Error>;

    fn reset(&mut self);
}

pub struct ColumnTreeBuilder<'a, ColumnArity, TreeArity>
where
    ColumnArity: Arity<Fr>,
    TreeArity: Arity<Fr>,
{
    pub leaf_count: usize,
    data: &'a mut [Fr],
    /// Index of the first unfilled datum.
    fill_index: usize,
    column_constants: PoseidonConstants<Bls12, ColumnArity>,
    pub column_batcher: Option<Batcher<ColumnArity>>,
    tree_builder: TreeBuilder<'a, TreeArity>,
}

impl<'a, ColumnArity, TreeArity> ColumnTreeBuilderTrait<ColumnArity, TreeArity>
    for ColumnTreeBuilder<'a, ColumnArity, TreeArity>
where
    ColumnArity: Arity<Fr>,
    TreeArity: Arity<Fr>,
{
    fn add_columns(&mut self, columns: &[GenericArray<Fr, ColumnArity>]) -> Result<(), Error> {
        let start = self.fill_index;
        let column_count = columns.len();
        let end = start + column_count;

        if end > self.leaf_count {
            return Err(Error::Other("too many columns".to_string()));
        }

        match self.column_batcher {
            Some(ref mut batcher) => {
                batcher.hash_into_slice(&mut self.data[start..start + column_count], columns)?;
            }
            None => columns.iter().enumerate().for_each(|(i, column)| {
                self.data[start + i] =
                    Poseidon::new_with_preimage(&column, &self.column_constants).hash();
            }),
        };

        self.fill_index += column_count;

        Ok(())
    }

    fn add_final_columns(
        &mut self,
        columns: &[GenericArray<Fr, ColumnArity>],
    ) -> Result<(Vec<Fr>, Vec<Fr>), Error> {
        self.add_columns(columns)?;

        let (base, tree) = self.tree_builder.add_final_leaves2(&mut self.data, self.fill_index)?;
        self.reset();

        Ok((base, tree))
    }

    fn reset(&mut self) {
        self.fill_index = 0;
        //self.data.iter_mut().for_each(|place| *place = Fr::zero());
    }
}
fn as_generic_arrays<'a, A: Arity<Fr>>(vec: &'a [Fr]) -> &'a [GenericArray<Fr, A>] {
    // It is a programmer error to call `as_generic_arrays` on a vector whose underlying data cannot be divided
    // into an even number of `GenericArray<Fr, Arity>`.
    assert_eq!(
        0,
        (vec.len() * std::mem::size_of::<Fr>()) % std::mem::size_of::<GenericArray<Fr, A>>()
    );

    // This block does not affect the underlying `Fr`s. It just groups them into `GenericArray`s of length `Arity`.
    // We know by the assertion above that `vec` can be evenly divided into these units.
    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const () as *const GenericArray<Fr, A>,
            vec.len() / A::to_usize(),
        )
    }
}

impl<'a, ColumnArity, TreeArity> ColumnTreeBuilder<'a, ColumnArity, TreeArity>
where
    ColumnArity: Arity<Fr>,
    TreeArity: Arity<Fr>,
{
    pub fn new(
        t: Option<BatcherType>,
        leaf_count: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
        data_buf:&'a mut [Fr]
    ) -> Result<Self, Error> {
        let column_batcher = match &t {
            Some(t) => Some(Batcher::<ColumnArity>::new(t, max_column_batch_size)?),
            None => None,
        };

        let tree_builder = match {
            match &column_batcher {
                #[cfg(feature = "gpu")]
                Some(b) => b.futhark_context(),
                #[cfg(feature = "opencl")]
                Some(b) => b.device(),
                None => None,
            }
        } {
            #[cfg(feature = "gpu")]
            Some(ctx) => TreeBuilder::<TreeArity>::new(
                Some(BatcherType::FromFutharkContext(ctx)),
                leaf_count,
                max_tree_batch_size,
                0,
            )?,
            #[cfg(feature = "opencl")]
            Some(device) => TreeBuilder::<TreeArity>::new(
                Some(BatcherType::FromDevice(device)),
                leaf_count,
                max_tree_batch_size,
                0,
                None,
            )?,
            None => TreeBuilder::<TreeArity>::new(t, leaf_count, max_tree_batch_size, 0, None)?,
        };

        info!("column_tree_builder data_buf len:{}, leaf count:{}", data_buf.len(), leaf_count);

        let builder = Self {
            leaf_count,
            data: data_buf,
            fill_index: 0,
            column_constants: PoseidonConstants::<Bls12, ColumnArity>::new(),
            column_batcher,
            tree_builder,
        };

        Ok(builder)
    }

    pub fn tree_size(&self) -> usize {
        self.tree_builder.tree_size(0)
    }

    pub fn tree_size2(leaf_count:usize) -> usize {
        TreeBuilder::<TreeArity>::tree_size2(leaf_count, 0)
    }

    // Compute root of tree composed of all identical columns. For use in checking correctness of GPU column tree-building
    // without the cost of generating a full column tree.
    pub fn compute_uniform_tree_root(
        &mut self,
        column: GenericArray<Fr, ColumnArity>,
    ) -> Result<Fr, Error> {
        // All the leaves will be the same.
        let element = Poseidon::new_with_preimage(&column, &self.column_constants).hash();

        self.tree_builder.compute_uniform_tree_root(element)
    }
}

#[cfg(all(any(feature = "gpu", feature = "opencl"), not(target_os = "macos")))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::Poseidon;
    use crate::BatchHasher;
    use bellperson::bls::Fr;
    use ff::Field;
    use generic_array::sequence::GenericSequence;
    use generic_array::typenum::{U11, U8};

    #[test]
    fn test_column_tree_builder() {
        // 16KiB tree has 512 leaves.
        test_column_tree_builder_aux(None, 512, 32, 512, 512);
        test_column_tree_builder_aux(Some(BatcherType::CPU), 512, 32, 512, 512);

        #[cfg(feature = "gpu")]
        test_column_tree_builder_aux(Some(BatcherType::GPU), 512, 32, 512, 512);

        #[cfg(feature = "opencl")]
        test_column_tree_builder_aux(Some(BatcherType::OpenCL), 512, 32, 512, 512);
    }

    fn test_column_tree_builder_aux(
        batcher_type: Option<BatcherType>,
        leaves: usize,
        num_batches: usize,
        max_column_batch_size: usize,
        max_tree_batch_size: usize,
    ) {
        let batch_size = leaves / num_batches;
        let mut data_buf = vec![Fr::defualt(); 0];
        let mut builder = ColumnTreeBuilder::<U11, U8>::new(
            batcher_type,
            leaves,
            max_column_batch_size,
            max_tree_batch_size,
            &mut data_buf,
        )
        .unwrap();

        // Simplify computing the expected root.
        let constant_element = Fr::zero();
        let constant_column = GenericArray::<Fr, U11>::generate(|_| constant_element);

        let max_batch_size = if let Some(batcher) = &builder.column_batcher {
            batcher.max_batch_size()
        } else {
            leaves
        };

        let effective_batch_size = usize::min(batch_size, max_batch_size);

        let mut total_columns = 0;
        while total_columns + effective_batch_size < leaves {
            let columns: Vec<GenericArray<Fr, U11>> =
                (0..effective_batch_size).map(|_| constant_column).collect();

            let _ = builder.add_columns(columns.as_slice()).unwrap();
            total_columns += columns.len();
        }

        let final_columns: Vec<_> = (0..leaves - total_columns)
            .map(|_| GenericArray::<Fr, U11>::generate(|_| constant_element))
            .collect();

        let (base, res) = builder.add_final_columns(final_columns.as_slice()).unwrap();

        let column_hash =
            Poseidon::new_with_preimage(&constant_column, &builder.column_constants).hash();
        assert!(base.iter().all(|x| *x == column_hash));

        let computed_root = res[res.len() - 1];

        let expected_root = builder.compute_uniform_tree_root(final_columns[0]).unwrap();
        let expected_size = builder.tree_builder.tree_size(0);

        assert_eq!(leaves, base.len());
        assert_eq!(expected_size, res.len());
        assert_eq!(expected_root, computed_root);
    }
}
