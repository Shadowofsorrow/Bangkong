import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from bangkong.pre_intelligent.reasoning_organs.reasoning_heads import ReasoningOrgans
from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
from bangkong.pre_intelligent.energy_layer.energy_consistency import GlobalConsistencyLayer

class BangkongConfig(GPT2Config):
    """
    Configuration class for Bangkong models.
    Inherits from GPT2Config to maintain compatibility with Hugging Face ecosystem.
    """
    model_type = "bangkong"

    def __init__(self, pi_latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        # Default PI latent dim to the model's hidden size (n_embd) if not explicitly set
        self.pi_latent_dim = pi_latent_dim if pi_latent_dim is not None else self.n_embd


class BangkongGPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    """
    Native Bangkong Causal Language Model.
    This architecture fully integrates Pre-Intelligent components (Reasoning, Memory, Consistency)
    """
    config_class = BangkongConfig

    def __init__(self, config):
        super().__init__(config)
        
        # 1. Base Transformer (GPT-2)
        self.transformer = GPT2Model(config)
        
        # 2. LM Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 3. Native Pre-Intelligent Components
        # We initialize them directly as modules so PyTorch registers their weights
        self.reasoning_organs = ReasoningOrgans(hidden_size=config.n_embd)
        
        # Memory system size configuration (can be parameterized in BangkongConfig later if needed)
        self.memory_system = HierarchicalMemory(
            hidden_size=config.n_embd,
            scratchpad_size=64,
            context_size=128,
            semantic_size=256
        )
        
        self.consistency_layer = GlobalConsistencyLayer(
            hidden_size=config.n_embd,
            energy_dim=256
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        # We force output_hidden_states to True so PI components can process them
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        enhancement_residual = None

        # 1. Apply Reasoning Organs
        try:
            ro_out = self.reasoning_organs(hidden_states)
            enhanced = ro_out.get("enhanced_hidden") if isinstance(ro_out, dict) else ro_out
            if enhanced is not None and enhanced.shape == hidden_states.shape:
                enhancement_residual = torch.zeros_like(hidden_states) if enhancement_residual is None else enhancement_residual
                enhancement_residual = enhancement_residual + 0.1 * enhanced
        except Exception:
            pass

        # 2. Apply Memory System
        try:
            mem_out = self.memory_system(hidden_states)
            enhanced = mem_out.get("enhanced", mem_out.get("output", None)) if isinstance(mem_out, dict) else mem_out
            if enhanced is not None and enhanced.shape == hidden_states.shape:
                enhancement_residual = torch.zeros_like(hidden_states) if enhancement_residual is None else enhancement_residual
                enhancement_residual = enhancement_residual + 0.1 * enhanced
        except Exception:
            pass

        # 3. Apply Consistency Layer
        try:
            cl_out = self.consistency_layer(hidden_states)
            enhanced = cl_out.get("enhanced_states") if isinstance(cl_out, dict) else cl_out
            if enhanced is not None and enhanced.shape == hidden_states.shape:
                enhancement_residual = torch.zeros_like(hidden_states) if enhancement_residual is None else enhancement_residual
                enhancement_residual = enhancement_residual + 0.1 * enhanced
        except Exception:
            pass

        # Safely integrate residual into hidden states before the LM head
        if enhancement_residual is not None:
            hidden_states = hidden_states + 0.1 * enhancement_residual
            
            # Update the stored hidden_states tuple so downstream tasks see the enhanced version
            if transformer_outputs.hidden_states is not None:
                new_states = list(transformer_outputs.hidden_states)
                new_states[-1] = hidden_states
                transformer_outputs.hidden_states = tuple(new_states)

        # Compute Logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

# Mendaftarkan kelas ke dalam AutoClass Hugging Face
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("bangkong", BangkongConfig)
AutoModelForCausalLM.register(BangkongConfig, BangkongGPT2LMHeadModel)
