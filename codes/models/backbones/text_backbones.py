import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import re
import numpy as np
from codes.models.backbones.text_basicblocks import Transformer, LayerNorm
from codes.registry import MODELS

@MODELS.register_module(name='text_backbones/Word2VecEncoder_Linear')
class Word2VecEncoder_Linear(nn.Module):
    def __init__(self,
                 embd_dim,
                 token_to_word_path,
                 num_embeddings=66250,
                 word_embedding_dim=300, # 300
                 word2vec_path='',
                 max_words=None,
                 output_dim=2048,
                 pooling='max'):
        super(Word2VecEncoder_Linear, self).__init__()
        self.word2vec_path = word2vec_path
        if self.word2vec_path is not None:
            self.word_embd = nn.Embedding.from_pretrained(torch.load(word2vec_path)) 

        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.pooling = pooling

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, x, raw_text=False):
        if self.word2vec_path:
            # Using word2vec
            if raw_text:
                x = self.words_to_ids(x).cuda()
            with torch.no_grad():
                x = self.word_embd(x)
            x = F.relu(self.fc1(x.float()), inplace=True)
        else:
            x = F.relu(self.fc1(x.float()), inplace=True)
        if self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'mean':
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError
        x = self.fc2(x)
        return x

@MODELS.register_module(name='text_backbones/Word2VecEncoder_Transformer')
class Sentence_Embedding_Transformer(nn.Module):
    def __init__(self,
                 embd_dim,
                 token_to_word_path,
                 num_embeddings=66250,
                 word_embedding_dim=300, # 300
                 word2vec_path='',
                 max_words=77,
                 output_dim=2048,
                 pooling='max',
                 # text
                 transformer_width=300,
                 transformer_layers=12,
                 transformer_heads=6):
        super(Sentence_Embedding_Transformer, self).__init__()
        self.context_length = max_words
        self.word2vec_path = word2vec_path
        if self.word2vec_path is not None:
            self.word_embd = nn.Embedding.from_pretrained(torch.load(word2vec_path)) 
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embd_dim))
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            return_intermediate=False,
        )
        self.ln_final = LayerNorm(transformer_width)

        self.initialize_parameters()

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return torch.stack(split_x, dim=0)


    def encode_text_light(self, x, raw_text):
        if self.word2vec_path:
            # Using word2vec
            if raw_text:
                x = self.words_to_ids(x).cuda()
            x = self.word_embd(x)   # [batch_size, n_ctx, d_model]
        return x


    def encode_text(self, xlight, text):
        x = xlight + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text, raw_text=False):
        text_lightemb = self.encode_text_light(text, raw_text)
        text_features = self.encode_text(text_lightemb, text)
        
        return text_features

@MODELS.register_module(name='text_backbones/BertEncoder')
class BertEncoder(nn.Module):
    def __init__(self, text_bert_type, text_last_n_layers, 
                text_aggregate_method, text_norm, text_embedding_dim, text_freeze_bert, text_agg_tokens):
        super(BertEncoder, self).__init__()

        self.bert_type = text_bert_type
        self.last_n_layers = text_last_n_layers
        self.aggregate_method = text_aggregate_method
        self.norm = text_norm
        self.embedding_dim = text_embedding_dim
        self.freeze_bert = text_freeze_bert
        self.agg_tokens = text_agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids=None, attn_mask=None, token_type=None, input_embedding=None):

        if input_embedding is not None:
            outputs = self.model(inputs_embeds=input_embedding, attention_mask=attn_mask)
        else:
            outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            # outputs: ['last_hidden_state', 'pooler_output', 'hidden_states']
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sents = None

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents




@MODELS.register_module(name='text_backbones/BertEncoder_embedding_layer')
class BertEncoder_embedding_layer(nn.Module):
    def __init__(self, text_bert_type, text_last_n_layers, 
                text_aggregate_method, text_norm, text_embedding_dim, text_freeze_bert, text_agg_tokens):
        super(BertEncoder_embedding_layer, self).__init__()

        self.bert_type = text_bert_type
        self.last_n_layers = text_last_n_layers
        self.aggregate_method = text_aggregate_method
        self.norm = text_norm
        self.embedding_dim = text_embedding_dim
        self.freeze_bert = text_freeze_bert
        self.agg_tokens = text_agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.text_projection = nn.Parameter(torch.empty(768, self.embedding_dim))
        
        self.initialize_parameters()

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False



    def aggregate_tokens(self, embeddings, caption_ids):

        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences


    def initialize_parameters(self):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=768** -0.5)

    def forward(self, ids=None, attn_mask=None, token_type=None, input_embedding=None):

        if input_embedding is not None:
            outputs = self.model(inputs_embeds=input_embedding, attention_mask=attn_mask)
        else:
            outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            # outputs: ['last_hidden_state', 'pooler_output', 'hidden_states']
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sents = None

        word_embeddings = word_embeddings @ self.text_projection # (bs, N, feat_dim --> emb_dim)

        sent_embeddings = sent_embeddings @ self.text_projection

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)

            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents


@MODELS.register_module(name='text_backbones/Word2Vec_Linear')
class Sentence_Embedding(nn.Module):
    def __init__(self,
                 embd_dim,
                 token_to_word_path,
                 num_embeddings=66250,
                 word_embedding_dim=300, # 300
                 word2vec_path='',
                 max_words=None,
                 output_dim=2048,
                 pooling='max'):
        super(Sentence_Embedding, self).__init__()
        self.word2vec_path = word2vec_path
        if self.word2vec_path is not None:
            self.word_embd = nn.Embedding.from_pretrained(torch.load(word2vec_path)) 

        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.pooling = pooling

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words).long()

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return torch.stack(split_x, dim=0)

    def forward(self, x, raw_text=False):
        if self.word2vec_path:
            # Using word2vec
            if raw_text:
                x = self.words_to_ids(x).cuda()
            with torch.no_grad():
                x = self.word_embd(x)
            x = F.relu(self.fc1(x.float()), inplace=True)
        else:
            x = F.relu(self.fc1(x.float()), inplace=True)
        if self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'mean':
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError
        x = self.fc2(x)
        return x