"""
This file is rewritten based on openprompt.plms.__init__.py with a small modifications
on model class initialization.
We write this file just to avoid direct coding on openprompt source codes.
"""

import math
from typing import List, Optional
from collections import namedtuple
from yacs.config import CfgNode
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertConfig, BertTokenizer, BertLMHeadModel,\
                         RobertaConfig, RobertaTokenizer, RobertaForCausalLM, \
                         AlbertTokenizer, AlbertConfig, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, \
                         OPTConfig, \
                         ElectraConfig, ElectraForMaskedLM, ElectraTokenizer, \
                         GPTJConfig, GPTJForCausalLM, \
                         LlamaConfig, LlamaTokenizer, LlamaModel, LlamaTokenizerFast, \
                         MistralConfig, PreTrainedTokenizerFast , AutoTokenizer


# from plm_special.models.gpt2 import GPT2Model
# from plm_special.models.llama import LlamaModel
# from plm_special.models.llama3 import LlamaModel as Llama3_2Model
# from plm_special.models.mistral import MistralModel
# from plm_special.models.opt import OPTModel
# from plm_special.models.t5 import T5Model


from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.t5.modeling_t5 import T5Model
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.opt.modeling_opt import OPTModel
from transformers.models.mistral.modeling_mistral import MistralModel

from plm_special.models.llm_models.t5 import T5Model
                        
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))

# The ** operator is used for unpacking a dictionary and passing it as keyword arguments to a function or class. In this case, it's unpacking the dictionary

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertLMHeadModel,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaForCausalLM
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2Model,
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
    }),
    't5-lm':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5Model,
    }),
    'opt': ModelClass(**{
        'config': OPTConfig,
        'tokenizer': GPT2Tokenizer,
        'model': OPTModel,
    }),
    'electra': ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model': ElectraForMaskedLM,
    }),
    "gptj": ModelClass(**{
        "config": GPTJConfig, 
        "tokenizer": GPT2Tokenizer, 
        "model": GPTJForCausalLM,
    }),
    "llama2": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel,
    }),
    "llama3": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": PreTrainedTokenizerFast,
        "model": LlamaModel,
    }),
    "mistral": ModelClass(**{
        "config": MistralConfig,
        "tokenizer": LlamaTokenizerFast,
        "model": MistralModel,
    }),
    "deepseek": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizerFast,
        "model": LlamaModel,
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def create_device_map(device_input_side: str, device_output_side: str, device_middle_side: str = None, hidden_layers = 32):
    """
    Create device map for any model. The device map is used to evenly split the Llama model into two/three parts on multiple devices.
    :param device_input_side: The device for the split of the model that receives the input (e.g., 'cuda:0').
    :param device_output_side: The device for the split of the model that produces the output (e.g., 'cuda:1').
    :param device_middle_side: The device for the split of the model that lies in the middle (e.g., 'cuda:2').
    :return: A device map dictionary.
    """
    device_map = {
        'embed_tokens': device_input_side  # Embedding layer on the input device
    }

    # Determine the device list based on whether a middle device is provided
    if device_middle_side is None:
        device_list = [device_input_side, device_output_side]
    else:
        device_list = [device_input_side, device_middle_side, device_output_side]

    # DeepSeek R1 - Llama3 distrill) has 32 transformer blocks
    num_layers = hidden_layers
    for i in range(num_layers):
        # Distribute layers evenly across devices
        device_map[f'layers.{i}'] = device_list[i // math.ceil(num_layers / len(device_list))]

    # Final normalization layer on the output device
    device_map['norm'] = device_output_side

    return device_map


def load_plm(model_name, model_path, specials_to_add = None, **kwargs):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ['<pad>']
    if 'llama' in model_name:
        specials_to_add = ['<pad>']
    if 'deepseek' in model_name:
        specials_to_add = ['<pad>']
    if 'bert' in model_name and 'roberta' not in model_name: # add is_decoder=True for BERT
        model_config.is_decoder = True
    if 'roberta' in model_name:  # add is_decoder=True for RoBERTa
        model_config.is_decoder = True

    # model = model_class.model.from_pretrained(model_path)
    device_input_side = kwargs.pop('device_input_side', None)
    device_output_side = kwargs.pop('device_output_side', None)
    if 'llama3' in model_name:
        print("-()*-()*"*10)
        print("Llama3 model")
        print("-()*-()*"*10)
    if 'deepseek' in model_name:
        print("-()*-()*"*10)
        print("deepseek model")
        print("-()*-()*"*10)
    
    if ('llama' in model_name or 'deepseek' in model_name) and device_input_side is not None and device_output_side is not None:
        print("Llama model Layers",model_config.num_hidden_layers)
        device_middle_side = kwargs.pop('device_middle_side', None)
        device_map = create_device_map(device_input_side, device_output_side, device_middle_side,hidden_layers=model_config.num_hidden_layers)
        model = model_class.model.from_pretrained(model_path, config=model_config, device_map=device_map)
    else:
        print("device_map flase else")
        model = model_class.model.from_pretrained(model_path, config=model_config)
    
    tokenizer = model_class.tokenizer.from_pretrained(model_path) 
    print("If tokenizer is loaded: ",tokenizer.encode("hello world"),"\n")



    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    
    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config


def load_plm_from_config(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    # you can change huggingface model_config here
    if 'gpt' in plm_config.model_name: # add pad token for gpt
        if "<pad>" not in config.plm.specials_to_add:
            config.plm.specials_to_add.append("<pad>")
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config


def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                print("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer
