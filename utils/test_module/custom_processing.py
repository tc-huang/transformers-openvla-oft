from transformers_openvla_oft import ProcessorMixin


class CustomProcessor(ProcessorMixin):
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
