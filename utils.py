import re
from typing import List, Optional

import kenlm
from pyannote.audio.pipelines import VoiceActivityDetection
from pyctcdecode import Alphabet, BeamSearchDecoderCTC
from pyctcdecode.alphabet import verify_alphabet_coverage
from pyctcdecode.constants import *
from pyctcdecode.language_model import LanguageModel, MultiLanguageModel
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM

from constants import *


def remove_unwanted_chars_and_uppercase(string):
    string = re.sub(REMOVE_BOLD, "", string)
    string = re.sub(CHARS_TO_IGNORE, "", string.upper())
    return " ".join(re.sub(CHARS_TO_BLANK, " ", string).split())


def build_processor(model_name, language_models):
    kenlm_model_paths = []
    unigram_file_paths = []
    alphas = []
    betas = []
    for lm in language_models:
        kenlm_model_paths.append(f'language_models/{lm["name"]}/ngram.bin')
        unigram_file_paths.append(f'language_models/{lm["name"]}/unigrams.txt')
        alphas.append(lm["alpha"])
        betas.append(lm["beta"])

    processor_without_lm = AutoProcessor.from_pretrained(model_name)

    vocab_dict = processor_without_lm.tokenizer.get_vocab()
    sorted_vocab_dict = {
        k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
    }
    labels = list(sorted_vocab_dict.keys())

    decoder = custom_build_ctcdecoder(
        labels=labels,
        kenlm_model_paths=kenlm_model_paths,
        unigram_file_paths=unigram_file_paths,
        alphas=alphas,
        betas=betas,
    )

    return Wav2Vec2ProcessorWithLM(
        feature_extractor=processor_without_lm.feature_extractor,
        tokenizer=processor_without_lm.tokenizer,
        decoder=decoder,
    )


def custom_build_ctcdecoder(
    labels: List[str],
    kenlm_model_paths: Optional[str] = None,
    unigram_file_paths: Optional[str] = None,
    alphas: List[float] = [],
    betas: List[float] = [],
    unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
    lm_score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
) -> BeamSearchDecoderCTC:
    """
    Customized version of build_ctcdecoder to enable multiple LMs.
    """
    alphabet = Alphabet.build_alphabet(labels)
    language_models = []

    for i, kenlm_model_path in enumerate(kenlm_model_paths):
        kenlm_model = kenlm.Model(kenlm_model_path)

        unigrams = open(unigram_file_paths[i], "r").readlines()
        unigrams = [x[:-1] for x in unigrams]  # Remove \n

        language_models.append(
            LanguageModel(
                kenlm_model,
                unigrams,
                alpha=alphas[i],
                beta=betas[i],
                unk_score_offset=unk_score_offset,
                score_boundary=lm_score_boundary,
            )
        )
        verify_alphabet_coverage(alphabet, unigrams[i])

    if len(language_models) > 1:
        language_model = MultiLanguageModel(language_models)
    else:
        language_model = language_models[0]

    return BeamSearchDecoderCTC(alphabet, language_model)


def extract_speak_segments(audio_path):
    # Function for defining ranges of active speech
    vad_pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

    vad_pipeline.instantiate(
        {
            # onset/offset activation thresholds
            "onset": 0.5,
            "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0,
        }
    )
    vad = vad_pipeline(audio_path)
    return vad.for_json()
