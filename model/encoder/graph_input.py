#coding=utf8
import os, math
import torch
import torch.nn as nn
from model.model_utils import rnn_wrapper, lens2mask, PoolingFunction
from utils.example import Example
from transformers import AutoModel, AutoConfig

class GraphInputLayer(nn.Module):

    def __init__(self, embed_size, hidden_size, word_vocab, dropout=0.2, fix_grad_idx=60, schema_aggregation='head+tail'):
        super(GraphInputLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_vocab = word_vocab
        self.fix_grad_idx = fix_grad_idx
        self.word_embed = nn.Embedding(self.word_vocab, self.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.embed_size, self.hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        self.word_embed.weight.grad[0].zero_() # padding symbol is always 0
        if index is not None:
            if not torch.is_tensor(index):
                index = torch.tensor(index, dtype=torch.long, device=self.word_embed.weight.grad.device)
            self.word_embed.weight.grad.index_fill_(0, index, 0.)
        else:
            self.word_embed.weight.grad[self.fix_grad_idx:].zero_()

    def forward(self, batch):
        question, table, column = self.word_embed(batch.questions), self.word_embed(batch.tables), self.word_embed(batch.columns)
        if batch.question_unk_mask is not None:
            question = question.masked_scatter_(batch.question_unk_mask.unsqueeze(-1), batch.question_unk_embeddings[:, :self.embed_size])
        if batch.table_unk_mask is not None:
            table = table.masked_scatter_(batch.table_unk_mask.unsqueeze(-1), batch.table_unk_embeddings[:, :self.embed_size])
        if batch.column_unk_mask is not None:
            column = column.masked_scatter_(batch.column_unk_mask.unsqueeze(-1), batch.column_unk_embeddings[:, :self.embed_size])
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs

class GraphInputLayerPLM(nn.Module):

    def __init__(self, plm='bert-base-uncased', hidden_size=256, dropout=0., subword_aggregation='mean',
            schema_aggregation='head+tail', lazy_load=False):
        super(GraphInputLayerPLM, self).__init__()
        self.plm_model = AutoModel.from_pretrained(plm) \
            if lazy_load else AutoModel.from_pretrained(plm)
        self.config = self.plm_model.config
        self.subword_aggregation = SubwordAggregation(self.config.hidden_size, subword_aggregation=subword_aggregation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.config.hidden_size, hidden_size, cell='lstm', schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        pass

    def forward(self, batch):
        outputs = self.plm_model(batch.inputs["input_ids"], attention_mask=batch.inputs["attention_mask"])[0] # final layer hidden states
        question, table, column = self.subword_aggregation(outputs, batch, self.plm_model)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs

class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, batch, plm_model):
        """ Transform pretrained model outputs into our desired format
        questions: bsize x max_question_len x hidden_size
        tables: bsize x max_table_word_len x hidden_size
        columns: bsize x max_column_word_len x hidden_size
        """
            
        # old_questions, old_tables, old_columns = inputs.masked_select(batch.question_mask_plm.unsqueeze(-1)), \
        #     inputs.masked_select(batch.table_mask_plm.unsqueeze(-1)), inputs.masked_select(batch.column_mask_plm.unsqueeze(-1))
        # # old questions shape = torch.Size([96256]) bao gồm tất cả questions
        # questions = old_questions.new_zeros(batch.question_subword_lens.size(0), batch.max_question_subword_len, self.hidden_size)
        # # questions = shape = torch.Size([62, 3, 1024]), mỗi question subword len = 62 (tổng độ dài các từ đã encode), max question subword len = 3
        # questions = questions.masked_scatter_(batch.question_subword_mask.unsqueeze(-1), old_questions)
        
        # tables = old_tables.new_zeros(batch.table_subword_lens.size(0), batch.max_table_subword_len, self.hidden_size)
        # tables = tables.masked_scatter_(batch.table_subword_mask.unsqueeze(-1), old_tables)
        # columns = old_columns.new_zeros(batch.column_subword_lens.size(0), batch.max_column_subword_len, self.hidden_size)
        # columns = columns.masked_scatter_(batch.column_subword_mask.unsqueeze(-1), old_columns)

        # questions = self.aggregation(questions, mask=batch.question_subword_mask)
        # tables = self.aggregation(tables, mask=batch.table_subword_mask)
        # columns = self.aggregation(columns, mask=batch.column_subword_mask)

        # new_questions, new_tables, new_columns = questions.new_zeros(len(batch), batch.max_question_len, self.hidden_size),\
        #     tables.new_zeros(batch.table_word_mask.size(0), batch.max_table_word_len, self.hidden_size), \
        #         columns.new_zeros(batch.column_word_mask.size(0), batch.max_column_word_len, self.hidden_size)
        # new_questions = new_questions.masked_scatter_(batch.question_mask.unsqueeze(-1), questions)
        # new_tables = new_tables.masked_scatter_(batch.table_word_mask.unsqueeze(-1), tables)
        # new_columns = new_columns.masked_scatter_(batch.column_word_mask.unsqueeze(-1), columns)

        # if len(batch.inputs["li_long_seq"]) != 0:
        #     pad_idx = Example.tokenizer.pad_token_id

        #     max_len =  max(batch.inputs["input_ids"][0].shape[0]) # lấy max len của những câu <= 252
        #     input_question_ids, input_table_ids, input_column_ids = [], [], []
        #     attention_question_mask, attention_table_mask, attention_column_mask = [], [], [] 
        #     question_mask_plm, table_mask_plm, column_mask_plm = [], [], []
        #     for ex in batch.inputs["li_long_seq"]:
        #         input_question_ids.append(ex.question_id + [pad_idx] * (max_len - len(ex.question_id)))
        #         attention_question_mask.append([1] * len(ex.question_id) + [0] * (max_len - len(ex.question_id)))
        #         qmp = [0] + [1] * (len(ex.question_id) - 2) + [0]
        #         question_mask_plm.append(qmp + [0] * (max_len - len(qmp)))

        #         input_table_ids.append(ex.table_id + [pad_idx] * (max_len - len(ex.table_id)))
        #         attention_table_mask.append([1] * len(ex.table_id) + [0] * (max_len - len(ex.table_id)))
        #         tmp = [1] * len(ex.table_id)
        #         table_mask_plm.append(tmp + [0] * (max_len - len(tmp)))

        #         input_column_ids.append(ex.column_id + [pad_idx] * (max_len - len(ex.column_id)))
        #         attention_column_mask.append([1] * len(ex.column_id) + [0] * (max_len - len(ex.column_id)))
        #         cmp = [1] * (len(ex.column_id) - 1) + [0]
        #         column_mask_plm.append(cmp + [0] * (max_len - len(cmp)))

        #     input_question_ids = torch.tensor(input_question_ids, dtype=torch.long, device=batch.device)
        #     input_table_ids = torch.tensor(input_table_ids, dtype=torch.long, device=batch.device)
        #     input_column_ids = torch.tensor(input_column_ids, dtype=torch.long, device=batch.device)

        #     attention_question_mask = torch.tensor(attention_question_mask, dtype=torch.float, device=batch.device)
        #     attention_table_mask = torch.tensor(attention_table_mask, dtype=torch.float, device=batch.device)
        #     attention_column_mask = torch.tensor(attention_column_mask, dtype=torch.float, device=batch.device)

        #     question_pretrained = plm_model(input_question_ids, attention_mask=attention_question_mask)
        #     table_pretrained = plm_model(input_table_ids, attention_mask=attention_table_mask)
        #     column_pretrained = plm_model(input_column_ids, attention_mask=attention_column_mask)


        #     question_mask_plm = torch.tensor(question_mask_plm, dtype=torch.bool, device=batch.device)
        #     table_mask_plm = torch.tensor(table_mask_plm, dtype=torch.bool, device=batch.device)
        #     column_mask_plm = torch.tensor(column_mask_plm, dtype=torch.bool, device=batch.device)

        pad_idx = Example.tokenizer.pad_token_id

        for idx, ex in enumerate(batch.examples):

            # return new_questions, new_tables, new_columns
            print(f'len long seq = {len(batch.inputs["li_long_seq"])}')
            if idx in batch.inputs["li_long_seq"]:

                max_len =  batch.inputs["input_ids"][0].shape[0] # lấy max len của những câu <= 252
                input_question_ids, input_table_ids, input_column_ids = [], [], []
                attention_question_mask, attention_table_mask, attention_column_mask = [], [], [] 
                question_mask_plm, table_mask_plm, column_mask_plm = [], [], []
                print(f'len long seq = {len(ex.input_id)}')
                print(f'len question long seq = {len(ex.question_id)}')
                print(f'len table long seq = {len(ex.table_id)}')
                print(f'len column long seq = {len(ex.column_id)}')
                input_question_ids.append(ex.question_id + [pad_idx] * (max_len - len(ex.question_id)))
                attention_question_mask.append([1] * len(ex.question_id) + [0] * (max_len - len(ex.question_id)))
                qmp = [0] + [1] * (len(ex.question_id) - 2) + [0]
                question_mask_plm.append(qmp + [0] * (max_len - len(qmp)))

                input_table_ids.append(ex.table_id + [pad_idx] * (max_len - len(ex.table_id)))
                attention_table_mask.append([1] * len(ex.table_id) + [0] * (max_len - len(ex.table_id)))
                tmp = [1] * len(ex.table_id)
                table_mask_plm.append(tmp + [0] * (max_len - len(tmp)))

                input_column_ids.append(ex.column_id + [pad_idx] * (max_len - len(ex.column_id)))
                attention_column_mask.append([1] * len(ex.column_id) + [0] * (max_len - len(ex.column_id)))
                cmp = [1] * (len(ex.column_id) - 1) + [0]
                column_mask_plm.append(cmp + [0] * (max_len - len(cmp)))

                input_question_ids = torch.tensor(input_question_ids, dtype=torch.long, device=batch.device)
                input_table_ids = torch.tensor(input_table_ids, dtype=torch.long, device=batch.device)
                input_column_ids = torch.tensor(input_column_ids, dtype=torch.long, device=batch.device)

                print(f'input_question_ids shape = {input_question_ids.shape}')
                print(f'input_table_ids shape = {input_table_ids.shape}')
                print(f'input_column_ids shape = {input_column_ids.shape}')
                attention_question_mask = torch.tensor(attention_question_mask, dtype=torch.float, device=batch.device)
                attention_table_mask = torch.tensor(attention_table_mask, dtype=torch.float, device=batch.device)
                attention_column_mask = torch.tensor(attention_column_mask, dtype=torch.float, device=batch.device)

                question_pretrained = plm_model(input_question_ids, attention_mask=attention_question_mask)[0]
                table_pretrained = plm_model(input_table_ids, attention_mask=attention_table_mask)[0]
                column_pretrained = plm_model(input_column_ids, attention_mask=attention_column_mask)[0]

                print(f'question pretrained shape = {question_pretrained.shape}')
                print(f'table pretrained shape = {table_pretrained.shape}')
                print(f'column pretrained shape = {column_pretrained.shape}')

                question_mask_plm = torch.tensor(question_mask_plm, dtype=torch.bool, device=batch.device)
                table_mask_plm = torch.tensor(table_mask_plm, dtype=torch.bool, device=batch.device)
                column_mask_plm = torch.tensor(column_mask_plm, dtype=torch.bool, device=batch.device)
                print(f'question_mask_plm shape = {question_mask_plm.shape}')
            else:
                pos = batch.inputs["id_map"][idx]
                print(f'phobert output shape = {inputs.shape}')
                print(f'phobert output = {inputs}')
                print(f'phobert output idx = {pos}: {inputs[pos]}')
                # print(f'batch.question_mask_plm.unsqueeze shape = {batch.question_mask_plm[pos].unsqueeze(-1).shape}')
                old_questions, old_tables, old_columns = inputs[pos].masked_select(batch.question_mask_plm[pos].unsqueeze(-1)), \
                    inputs.masked_select(batch.table_mask_plm[pos].unsqueeze(-1)), inputs.masked_select(batch.column_mask_plm[pos].unsqueeze(-1))
                # print(f'old questions = {old_questions}')
                # print(f'old questions shape = {old_questions.shape}')
                
                questions = old_questions.new_zeros(batch.question_subword_lens[pos].size(0), batch.max_question_subword_len, self.hidden_size)
                # print(f'old_questions.new_zeros = {questions}')
                # print(f'old_questions.new_zeros shape = {questions.shape}')
                # print(f'batch.question_subword_mask.unsqueeze = {batch.question_subword_mask(pos).unsqueeze(-1)}')
                # print(f'batch.question_subword_mask.unsqueeze shape = {batch.question_subword_mask(pos).unsqueeze(-1).shape}')
                questions = questions.masked_scatter_(batch.question_subword_mask(pos).unsqueeze(-1), old_questions)
                # print(f'questions.masked_scatter = {questions}')
                # print(f'questions.masked_scatter shape = {questions.shape}')

                tables = old_tables.new_zeros(batch.table_subword_lens[pos].size(0), batch.max_table_subword_len, self.hidden_size)
                tables = tables.masked_scatter_(batch.table_subword_mask(pos).unsqueeze(-1), old_tables)
                
                columns = old_columns.new_zeros(batch.column_subword_lens[pos].size(0), batch.max_column_subword_len, self.hidden_size)
                columns = columns.masked_scatter_(batch.column_subword_mask(pos).unsqueeze(-1), old_columns)

                questions = self.aggregation(questions, mask=batch.question_subword_mask)
                tables = self.aggregation(tables, mask=batch.table_subword_mask)
                columns = self.aggregation(columns, mask=batch.column_subword_mask)

                new_questions, new_tables, new_columns = questions.new_zeros(len(batch), batch.max_question_len, self.hidden_size),\
                    tables.new_zeros(batch.table_word_mask.size(0), batch.max_table_word_len, self.hidden_size), \
                        columns.new_zeros(batch.column_word_mask.size(0), batch.max_column_word_len, self.hidden_size)
                new_questions = new_questions.masked_scatter_(batch.question_mask.unsqueeze(-1), questions)
                new_tables = new_tables.masked_scatter_(batch.table_word_mask.unsqueeze(-1), tables)
                new_columns = new_columns.masked_scatter_(batch.column_word_mask.unsqueeze(-1), columns)
        return new_questions, new_tables, new_columns
    

class InputRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell='lstm', schema_aggregation='head+tail', share_lstm=False):
        super(InputRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell.upper()
        self.question_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_lstm = self.question_lstm if share_lstm else \
            getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_aggregation = schema_aggregation
        if self.schema_aggregation != 'head+tail':
            self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=schema_aggregation)

    def forward(self, input_dict, batch):
        """
            for question sentence, forward into a bidirectional LSTM to get contextual info and sequential dependence
            for schema phrase, extract representation for each phrase by concatenating head+tail vectors,
            batch.question_lens, batch.table_word_lens, batch.column_word_lens are used
        """
        questions, _ = rnn_wrapper(self.question_lstm, input_dict['question'], batch.question_lens, cell=self.cell)
        questions = questions.contiguous().view(-1, self.hidden_size)[lens2mask(batch.question_lens).contiguous().view(-1)]
        table_outputs, table_hiddens = rnn_wrapper(self.schema_lstm, input_dict['table'], batch.table_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            tables = self.aggregation(table_outputs, mask=batch.table_word_mask)
        else:
            table_hiddens = table_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else table_hiddens.transpose(0, 1)
            tables = table_hiddens.contiguous().view(-1, self.hidden_size)
        column_outputs, column_hiddens = rnn_wrapper(self.schema_lstm, input_dict['column'], batch.column_word_lens, cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            columns = self.aggregation(column_outputs, mask=batch.column_word_mask)
        else:
            column_hiddens = column_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else column_hiddens.transpose(0, 1)
            columns = column_hiddens.contiguous().view(-1, self.hidden_size)

        questions = questions.split(batch.question_lens.tolist(), dim=0)
        tables = tables.split(batch.table_lens.tolist(), dim=0)
        columns = columns.split(batch.column_lens.tolist(), dim=0)
        # dgl graph node feats format: q11 q12 ... t11 t12 ... c11 c12 ... q21 q22 ...
        outputs = [th for q_t_c in zip(questions, tables, columns) for th in q_t_c]
        outputs = torch.cat(outputs, dim=0)
        # transformer input format: bsize x max([q1 q2 ... t1 t2 ... c1 c2 ...]) x hidden_size
        # outputs = []
        # for q, t, c in zip(questions, tables, columns):
        #     zero_paddings = q.new_zeros((batch.max_len - q.size(0) - t.size(0) - c.size(0), q.size(1)))
        #     cur_outputs = torch.cat([q, t, c, zero_paddings], dim=0)
        #     outputs.append(cur_outputs)
        # outputs = torch.stack(outputs, dim=0) # bsize x max_len x hidden_size
        return outputs
