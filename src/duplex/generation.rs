//! Moshi's delayed audio-codebook clock with externally selected words.
//!
//! The codec-delay mechanism follows Kyutai's Apache-2.0/MIT implementation:
//! https://github.com/kyutai-labs/moshi/blob/main/rust/moshi-core/src/lm_generate_multistream.rs
//! Copyright (c) Kyutai. Orchard keeps only the bounded delay ring and lets the
//! model choose text-pad timing; prescribing a word every fixed N frames can
//! cut syllables off even when the echoed text tokens look correct.
use super::{language::LanguageModel, DuplexOptions};
use candle_core::{IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

const UNGENERATED: u32 = u32::MAX;
const RING: usize = 8;

pub(super) struct State {
    model: LanguageModel,
    config: moshi::lm_generate_multistream::Config,
    audio: [[u32; 16]; RING],
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
    force_audio: moshi::lm::ForcedAudioTokens,
    step_idx: usize,
    frames_since_text: usize,
    max_text_wait: usize,
    prepend: Option<Tensor>,
    reference: Option<Tensor>,
    reference_index: usize,
}

pub(super) struct Step {
    pub text_token: u32,
    pub consumed_text: bool,
}

impl State {
    pub fn new(model: LanguageModel, options: &DuplexOptions, acoustic_delay: usize) -> Self {
        let mut config = moshi::lm_generate_multistream::Config::v0_1();
        config.acoustic_delay = acoustic_delay;
        let prepend = model.speaker_condition("user");
        let sampling = |temperature: f64, k: usize| {
            if temperature <= 0.0 {
                Sampling::ArgMax
            } else {
                Sampling::TopK { k, temperature }
            }
        };
        let force_audio = moshi::lm::ForcedAudioTokens::new(
            config.acoustic_delay,
            config.audio_pad_token(),
            &[8, 8],
        );
        Self {
            model,
            config,
            audio: [[UNGENERATED; 16]; RING],
            force_audio,
            audio_lp: LogitsProcessor::from_sampling(
                options.seed,
                sampling(options.audio_temperature, options.audio_top_k),
            ),
            text_lp: LogitsProcessor::from_sampling(
                options.seed.wrapping_add(1),
                sampling(options.text_temperature, options.text_top_k),
            ),
            step_idx: 0,
            frames_since_text: 0,
            max_text_wait: options.text_token_interval,
            prepend,
            reference: None,
            reference_index: 0,
        }
    }
    pub fn set_reference(&mut self, reference: Tensor) {
        self.reference = Some(reference);
        self.reference_index = 0;
    }

    pub fn reference_remaining(&self) -> usize {
        self.reference
            .as_ref()
            .and_then(|t| t.dim(1).ok())
            .unwrap_or(0)
            .saturating_sub(self.reference_index)
    }
    pub fn clear_reference(&mut self) {
        self.reference = None;
        self.reference_index = 0;
    }

    pub fn config(&self) -> &moshi::lm_generate_multistream::Config {
        &self.config
    }
    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    pub fn step(
        &mut self,
        previous_text: u32,
        input_audio: &[u32],
        requested_text: Option<u32>,
    ) -> candle_core::Result<Step> {
        if self.step_idx == 0 {
            if let Some(condition) = self.prepend.take() {
                self.model.prepend(&condition)?;
            }
        }
        if input_audio.len() != 8 {
            candle_core::bail!("Moshi expects eight input codebooks");
        }
        self.audio[self.step_idx % RING] = [UNGENERATED; 16];
        self.audio[self.step_idx % RING][8..].copy_from_slice(input_audio);
        let device = self.model.device();
        let pad = self.config.audio_pad_token();
        let mut codes = Vec::with_capacity(16);
        for codebook in 0..16 {
            let token = if codebook == 0 || codebook == 8 {
                if self.step_idx == 0 {
                    pad
                } else {
                    self.audio[(self.step_idx - 1) % RING][codebook]
                }
            } else if self.step_idx <= self.config.acoustic_delay {
                pad
            } else {
                self.audio[(self.step_idx - self.config.acoustic_delay - 1) % RING][codebook]
            };
            if token == UNGENERATED {
                candle_core::bail!("Ungenerated Moshi delayed codebook {codebook}");
            }
            codes.push(Some(Tensor::from_slice(&[token], (1, 1), device)?));
        }
        let previous_text = Some(Tensor::from_slice(&[previous_text], (1, 1), device)?);
        let reference = match self.reference.as_ref() {
            Some(tensor) if self.reference_index < tensor.dim(1)? => {
                let step = tensor.i((.., self.reference_index..self.reference_index + 1, ..))?;
                self.reference_index += 1;
                Some(step)
            }
            _ => {
                self.reference = None;
                None
            }
        };
        let (logits, hidden) = self.model.forward(previous_text, codes, reference)?;
        let (text_token, consumed_text) = if requested_text == Some(self.config.text_pad_token) {
            (self.config.text_pad_token, false)
        } else {
            let sampled = self.text_lp.sample(&logits.i((0, 0))?)?;
            match requested_text {
                None => (sampled, false),
                Some(wanted) => choose_text(
                    sampled,
                    wanted,
                    self.config.text_pad_token,
                    self.frames_since_text,
                    self.max_text_wait,
                ),
            }
        };
        if text_token == self.config.text_pad_token {
            self.frames_since_text += 1;
        } else {
            self.frames_since_text = 0;
        }
        let generated = self.model.sample_audio(
            &hidden,
            text_token,
            self.force_audio.forced_tokens(self.step_idx),
            &mut self.audio_lp,
        )?;
        for codebook in 0..8 {
            let delay = if codebook == 0 {
                0
            } else {
                self.config.acoustic_delay
            };
            let position = self.step_idx.saturating_sub(delay) % RING;
            self.audio[position][codebook] =
                generated.as_ref().map_or(pad, |tokens| tokens[codebook]);
        }
        self.step_idx += 1;
        Ok(Step {
            text_token,
            consumed_text,
        })
    }

    pub fn last_audio_tokens(&self) -> Option<Vec<u32>> {
        if self.step_idx <= self.config.acoustic_delay {
            return None;
        }
        let frame = &self.audio[(self.step_idx - self.config.acoustic_delay - 1) % RING];
        if frame
            .iter()
            .any(|token| *token >= self.config.audio_pad_token())
        {
            return None;
        }
        Some(frame[..8].to_vec())
    }
}

fn choose_text(sampled: u32, wanted: u32, pad: u32, waited: usize, max_wait: usize) -> (u32, bool) {
    if sampled == pad && waited < max_wait {
        (pad, false)
    } else {
        (wanted, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn natural_pad_timing_never_discards_the_backbones_waiting_word() {
        let mut words = std::collections::VecDeque::from([42, 43]);
        let mut rendered = Vec::new();
        let mut waited = 0;
        for prediction in [3, 3, 99, 3, 3, 3, 88] {
            let (token, consumed) = choose_text(prediction, *words.front().unwrap(), 3, waited, 12);
            if consumed {
                rendered.push(token);
                words.pop_front();
                waited = 0;
            } else {
                waited += 1;
            }
        }
        assert_eq!(rendered, [42, 43]);
        assert!(words.is_empty());
    }
}
