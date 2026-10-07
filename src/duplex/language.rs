//! Moshi language-model variants share the audio clock. MoshiRAG uses its
//! trained ARC reference channel instead of pretending a dialogue LM is TTS.
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

#[derive(Clone)]
pub(super) enum LanguageModel {
    Base(moshi::lm::LmModel),
    Rag(moshi_rag::lm::LmModel),
}
impl LanguageModel {
    pub fn is_rag(&self) -> bool {
        matches!(self, Self::Rag(_))
    }
    pub fn device(&self) -> &Device {
        match self {
            Self::Base(m) => m.device(),
            Self::Rag(m) => m.device(),
        }
    }
    pub fn reset_state(&mut self) {
        match self {
            Self::Base(m) => m.reset_state(),
            Self::Rag(m) => m.reset_state(),
        }
    }
    pub fn prepend(&mut self, condition: &Tensor) -> candle_core::Result<()> {
        match self {
            Self::Base(_) => candle_core::bail!("Base Moshi has no prepend conditioner"),
            Self::Rag(m) => {
                m.forward_prepend(condition, None)?;
                Ok(())
            }
        }
    }
    pub fn speaker_condition(&self, value: &str) -> Option<Tensor> {
        match self {
            Self::Base(_) => None,
            Self::Rag(m) => m.get_lut_condition("first_speaker", value),
        }
    }
    pub fn forward(
        &mut self,
        text: Option<Tensor>,
        codes: Vec<Option<Tensor>>,
        reference: Option<Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Self::Base(model) => {
                if reference.is_some() {
                    candle_core::bail!("Base Moshi cannot receive reference embeddings");
                }
                model.forward_cond(text, codes, None, &().into())
            }
            Self::Rag(model) => {
                let condition = reference.map(moshi_rag::conditioner::Condition::AddToInput);
                model.forward_cond(text, codes, condition.as_ref(), &().into())
            }
        }
    }
    pub fn sample_audio(
        &mut self,
        hidden: &Tensor,
        text: u32,
        forced: &[Option<u32>],
        sampler: &mut LogitsProcessor,
    ) -> candle_core::Result<Option<Vec<u32>>> {
        match self {
            Self::Base(model) => model.depformer_sample(hidden, Some(text), forced, sampler),
            Self::Rag(model) => model.depformer_sample(hidden, Some(text), forced, sampler),
        }
    }
}
