import sys
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions, )
from transformers.modeling_utils import Conv1D
from typing import Tuple

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        text_embed = hidden_states
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), {
            'image_embed': encoder_hidden_states,
            'text_embed': text_embed
        }


class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gama = None 
        self.beta = None
        self.alpha = None
        self.temp = None
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.semantic_input_proj = nn.Linear(config.n_embd, config.n_embd)  
        self.semantic_output_proj = nn.Linear(config.n_embd, config.n_embd)
        self.instance_input_proj = nn.Linear(config.n_embd, config.n_embd)
        self.instance_output_proj = nn.Linear(config.n_embd, config.n_embd)
        self.tau =  0.2
        self.neg_eps = 1.0  
        self.pos_eps = 3.0
        self.input_projection = nn.Linear(config.n_embd, config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)

        self.lossfn = CELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    
    def compute_k_least_similar_embeddings(self,embeddings, k):
        """
        Compute the K least similar embeddings for each sample in the batch.
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim].
            k: Number of least similar samples to retrieve for each sample.
        Returns:
            Tensor of shape [batch_size, k], containing indices of the least similar samples for each sample.
        """
        sim_matrix = embeddings @ embeddings.T  # Compute similarity matrix (cosine similarity)
        sim_matrix.fill_diagonal_(float('inf'))  # Exclude self-similarity by setting diagonal to infinity
        least_similar_indices = torch.topk(sim_matrix, k=k, largest=False, dim=-1).indices  # Find K least similar
        hard_negative_weight = torch.zeros_like(sim_matrix)  
        batch_size = embeddings.shape[0]
        for i in range(batch_size):
            hard_negative_weight[i, least_similar_indices[i]] = 1.0  

        return hard_negative_weight,least_similar_indices 

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,   
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ques_end=None,
            ans_end=None,
            cap_end=None,
            dec_mask=None,
            answer_mask=None,
            pos_patches = None,
            ques_mask=None,
            answer_idx=None,
            is_cam=0
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        encoder_hidden_states.requires_grad=True


        transformer_outputs, inputs = self.transformer(   
            input_ids = input_ids,
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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous() 
           
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  

            image_input = inputs['image_embed']
            text_input = inputs['text_embed']
            text_output = hidden_states
            batch_size = text_output.size()[0]

            # CCL
            #v,q,a
            img_ques_ans_embeddings = torch.cat(    
                [torch.mean(torch.cat([image_input[i], text_input[i, :ans_end[i], :]], dim=0), dim=0).unsqueeze(0) for i in range(0, batch_size)])
            #report
            rep_embeddings = torch.cat([torch.mean(text_output[i, ans_end[i]:cap_end[i], :], dim=0).unsqueeze(0) for i in range(0, batch_size)])   #[bs,dim]
            img_ques_ans_proj = F.normalize(self.semantic_input_proj(img_ques_ans_embeddings), dim=-1)
            rep_proj = F.normalize(self.semantic_output_proj(rep_embeddings), dim=-1)
            report_feats = torch.cat([torch.mean(text_input[i, ans_end[i]:cap_end[i], :], dim=0).unsqueeze(0) for i in range(0, batch_size)])
            rep_proj_select = F.normalize(report_feats, dim=-1)  
            hard_negative_weight,least_similar_indices  = self.compute_k_least_similar_embeddings(rep_proj_select,5)
            loss_sem = self.cl_loss(rep_proj,img_ques_ans_proj,hard_negative_weight)
            
            # RCL
            loss_ins=0
            if is_cam!=0:
                pred = torch.cat([shift_logits[i, ques_end[i] + 3, :].unsqueeze(0) for i in range(0, batch_size)]) 
                visual_grad = torch.autograd.grad((pred * answer_idx).sum(), image_input, create_graph=True)[0]  
                word_grad = torch.autograd.grad((pred * answer_idx).sum(), text_input, create_graph=True)[0]
        
                word_grad_cam = word_grad.sum(2)  
                word_grad_cam_sigmoid = word_grad_cam * ques_mask  
                w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :3]
                ques_mask.scatter_(1, w_ind, 0) 
                neg_text_input = text_input * ques_mask.unsqueeze(-1)  
                pos_ques_mask = torch.zeros_like(ques_mask)
                pos_ques_mask.scatter_(1, w_ind, 1)
                pos_text_input = text_input * pos_ques_mask.unsqueeze(-1)  

                # visual_cam
                visual_grad_cam = visual_grad.sum(2)
                v_ind = visual_grad_cam.sort(1, descending=True)[1][:, :30] 
                pos_visual_mask = torch.zeros(batch_size, image_input.size(1), device=self.device)
                neg_visual_mask = torch.ones_like(pos_visual_mask)
                pos_visual_mask.scatter_(1, v_ind, 1)
                neg_visual_mask.scatter_(1, v_ind, 0)  
                pos_visual_input = image_input * pos_visual_mask.unsqueeze(-1) 
                neg_visual_input = image_input * neg_visual_mask.unsqueeze(-1) 
                pos_visual_word = torch.cat(
                    [torch.mean(torch.cat([pos_visual_input[i], pos_text_input[i]], dim=0), dim=0).unsqueeze(0) for i in
                        range(0, batch_size)])
                pos_visual_word = F.normalize(self.instance_input_proj(pos_visual_word), dim=-1)  
                neg_visual_word = torch.cat(
                    [torch.mean(torch.cat([neg_visual_input[i], neg_text_input[i]], dim=0), dim=0).unsqueeze(0) for i in
                        range(0, batch_size)])
                neg_visual_word = F.normalize(self.instance_input_proj(neg_visual_word), dim=-1)

                cap_ans_embeddings = torch.cat(
                    [torch.mean(text_output[i, ques_end[i] : cap_end[i]-1 , :], dim=0).unsqueeze(0) for i in
                     range(0, batch_size)])
                cap_ans_proj = F.normalize(self.instance_output_proj(cap_ans_embeddings), dim=-1)

                loss_ins = self.info_criterion(pos_visual_word, neg_visual_word, cap_ans_proj)
            
                

            # APCL
            cos = nn.CosineSimilarity(dim=-1)

            proj_enc_h = self.input_projection(encoder_hidden_states)
            proj_dec_h = self.output_projection(text_output)

            attention_mask = torch.ones(encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], dtype=torch.long, device=input_ids.device)
            decoder_attention_mask = dec_mask  

            avg_doc = self.avg_pool(proj_enc_h, attention_mask)  
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)  

            # generate positive samples
            pos_dec_hidden = self.generate_cont_adv(
                encoder_hidden_states, attention_mask, text_output, decoder_attention_mask,
                lm_logits, self.tau, self.pos_eps
            )
            avg_pos_dec = self.avg_pool(self.output_projection(pos_dec_hidden), decoder_attention_mask)  

            # generate negative samples
            perturbed_dec = self.generate_adv(text_output, labels)  
            proj_pert_dec_h = self.output_projection(perturbed_dec)
            avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)  

            negative_samples = []
            for i in range(batch_size):
                negatives = avg_abs[least_similar_indices[i]]  
               
                negative_samples.append(negatives)
            negative_samples = torch.stack(negative_samples)  
           
            negative_samples = torch.cat([negative_samples, avg_pert.unsqueeze(1)], dim=1)  
           
            pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(1)  
            
            avg_doc_norm = F.normalize(avg_doc, p=2, dim=-1) 
            negative_samples_norm = F.normalize(negative_samples, p=2, dim=-1)  
            neg_sim = torch.einsum('bd,bkd->bk', avg_doc_norm, negative_samples_norm)  
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.tau  
            cl_labels = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device) 
            cont_loss = self.criterion(logits, cl_labels)
            
            loss += self.alpha * loss_sem  + self.beta * loss_ins + self.gama * cont_loss

        
            
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def get_end_idx(self,logits,eos_id,cap_end):
        predicted_ids = logits.argmax(dim=-1)  
        end_idx = []
        for i in range(0,logits.shape[0]):
            eos_positions = torch.where(predicted_ids[i] == eos_id)
            if eos_positions[0].numel() > 0:
                end_idx.append(eos_positions[0][-1].item())  #最后一个eos
            else:
                end_idx.append(cap_end[i])

    def cl_loss(self, x, y, score=None):
        device = x.device
        sim = x @ y.T / self.temp
        labels = torch.arange(sim.shape[0], device=device, dtype=torch.long)
        return self.lossfn(sim, labels, score)

    def set_config(self, config):
        self.temp = config.temp
        self.alpha = config.alpha
        self.beta = config.beta 
        self.gama = config.gama


    def info_criterion(self, pos_hiddens, neg_hiddens, anchor_hiddens):
        device = pos_hiddens.device
        # loss_p = pos_hiddens @ anchor_hiddens.T / self.temp
        # loss_n = neg_hiddens @ anchor_hiddens.T / self.temp
        loss_p = ((pos_hiddens * anchor_hiddens).sum(1))
        loss_n = ((neg_hiddens * anchor_hiddens).sum(1))

        logits = torch.cat([loss_p.unsqueeze(1), loss_n.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss_cl = self.criterion(logits, labels)
        return loss_cl

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def generate_adv(self,dec_hiddens, labels):
        dec_hiddens = dec_hiddens.detach()
        dec_hiddens.requires_grad = True
        dec_lm_logits = self.lm_head(dec_hiddens)  
        
        shift_logits = dec_lm_logits[..., :-1, :].contiguous()  
        shift_labels = labels[..., 1:].contiguous()  
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 

        dec_grad = torch.autograd.grad(loss, dec_hiddens)[0]  

        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()


        perturbed_dec = perturbed_dec 
        

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()   
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.input_projection(enc_hiddens),enc_mask) 
        avg_dec = self.avg_pool(self.output_projection(dec_hiddens),dec_mask) 

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),device=enc_hiddens.device)
        loss = cont_crit(logits, labels) 
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12) 

        perturb_dec_hidden = dec_hiddens + eps * dec_grad 

        perturb_dec_hidden = perturb_dec_hidden.detach()     
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.lm_head(perturb_dec_hidden) 

        true_probs = F.softmax(lm_logits, -1)  
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()  

        perturb_log_probs = F.log_softmax(perturb_logits, -1) 

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size), 
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad
    
        return perturb_dec_hidden


    def avg_pool(self, hidden_states, mask):  
        length = torch.sum(mask, 1, keepdim=True).float()    
        mask = mask.unsqueeze(2)   
        hidden = hidden_states.masked_fill(mask == 0, 0.0)   
        avg_hidden = torch.sum(hidden, 1) / length    

        return avg_hidden

class CELoss(nn.Module):
    """ Cross Entropy Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, hard_negative_weight=None):
        """
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        """
        eps = 1e-12

        # standard cross entropy loss
        if hard_negative_weight is not None:
            weight = hard_negative_weight.clone()
            weight[range(hard_negative_weight.size(0)), range(hard_negative_weight.size(0))] = 1

            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred * weight).sum(dim=1)+ eps)
        else:
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        return loss.mean()
