import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor


class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)


class SubInstructionEncoder(nn.Module):
    def __init__(self, config: Config):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                vocab_size: number of words in the vocabulary
                embedding_size: The dimension of each embedding vector
                use_pretrained_embeddings:
                embedding_file:
                fine_tune_embeddings:
                dataset_vocab:
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: Whether or not to return just the final state
        """
        super().__init__()

        self.config = config

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.bidir = config.bidirectional
        self.encoder_rnn_global = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=self.bidir,
            batch_first=True,
        )
        self.encoder_rnn_local = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=self.bidir,
            batch_first=True,
        )
        self.final_state_only = config.final_state_only
        self.use_sub_instruction = config.use_sub_instruction

    @property
    def output_size(self):
        return self.config.hidden_size * (2 if self.bidir else 1)

    def _load_embeddings(self):
        """Loads word embeddings from a pretrained embeddings file.

        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged:
            https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ

        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def _aggregate(self, sub_output, sub_lengths):
        # """ Aggregates embeddings in a sub instruction to get its representation
        # Args:
        #     sub_inst: [sub_seq_len, embedding_dim]
        # Return:
        #     [embedding_dim]
        # """
        # res = torch.mean(sub_inst, dim=0)
        """Aggregates embeddings in a sub instruction to get its representation
        Args:
            sub_output: (batch_size, max_sub_num, sub_seq_len, output_size)
            sub_lengths: (batch_size, max_sub_num, 1)
        Return:
            res: (batch_size, max_sub_num, output_size)
        """
        sub_lengths[sub_lengths == 0] = 1e-8
        res = torch.sum(sub_output, dim=2) / sub_lengths
        return res

    def forward(self, observations):
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = observations["instruction"].long()
        sub_instruction = observations["sub_instruction"].long()
        batch_size = instruction.size(0)

        lengths = (instruction != 0.0).long().sum(dim=1)
        embedded = self.embedding_layer(instruction)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, final_state = self.encoder_rnn_global(
            packed_seq
        )  # output: [batch_size x seq_len x hidden_size]

        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]
        if self.use_sub_instruction:  # use sub instructions
            sub_instruction_stack = torch.flatten(
                sub_instruction, start_dim=0, end_dim=1
            )
            sub_embedded_stack = self.embedding_layer(sub_instruction_stack)
            # get the number of sub instructions in each sample
            sub_num = (instruction == self.config.SPLIT_INDEX).sum(dim=1)
            # construct sub_ tensors
            sub_lengths = torch.argmax(
                (sub_instruction_stack == self.config.PAD_INDEX).int(), dim=1
            )
            zero_idx = sub_lengths == 0
            sub_lengths[sub_lengths == 0] = 1
            # encode sub instructions
            sub_packed_seq = nn.utils.rnn.pack_padded_sequence(
                sub_embedded_stack,
                sub_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            sub_output_stack, sub_final_state = self.encoder_rnn_local(sub_packed_seq)

            # post process
            if self.config.rnn_type == "LSTM":
                sub_final_state = sub_final_state[0]
            sub_output_stack = nn.utils.rnn.pad_packed_sequence(
                sub_output_stack,
                batch_first=True,
                total_length=self.config.SUB_PAD_SIZE,
            )[0]
            sub_output_stack[zero_idx, 0, :] = 0
            sub_output = sub_output_stack.reshape(
                (
                    batch_size,
                    self.config.MAX_SUB_NUM,
                    self.config.SUB_PAD_SIZE,
                    self.output_size,
                )
            )
            sub_lengths_divide = sub_lengths.reshape(
                (batch_size, self.config.MAX_SUB_NUM, 1)
            )
            sub_representation = self._aggregate(sub_output, sub_lengths_divide)
            sub_final_state = sub_final_state.squeeze(0)

            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            final_state = final_state.squeeze(0)

            # output: (batch_size, seq_len, output_size)
            # sub_output: (batch_size, max_sub_num, sub_seq_len, output_size)
            # sub_representation: (batch_size, max_sub_num, output_size)
            # sub_num: (batch_size)
            # sub_lengths: (total_size)
            if self.final_state_only:
                return (
                    final_state,
                    sub_final_state,
                    sub_representation,
                    sub_num,
                    sub_lengths,
                )
            else:
                return (
                    output.permute(0, 2, 1),
                    sub_output,
                    sub_representation,
                    sub_num,
                    sub_lengths,
                )
        else:
            if self.final_state_only:
                return final_state.squeeze(0)
            else:
                return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                    0
                ].permute(
                    0, 2, 1
                )  # the permutation is for Conv1d


## old version
# def forward(self, observations):
#     """
#     Tensor sizes after computation:
#         instruction: [batch_size x seq_length]
#         lengths: [batch_size]
#         hidden_state: [batch_size x hidden_size]
#     """
#     instruction = observations["instruction"].long()
#     sub_instruction = observations["sub_instruction"].long()
#     batch_size = instruction.size(0)
#     device = instruction.device

#     lengths = (instruction != 0.0).long().sum(dim=1)
#     embedded = self.embedding_layer(instruction)

#     packed_seq = nn.utils.rnn.pack_padded_sequence(
#         embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
#     )

#     output, final_state = self.encoder_rnn_global(packed_seq) # output: [batch_size x seq_len x hidden_size]

#     if self.config.rnn_type == "LSTM":
#         final_state = final_state[0]
#     if self.use_sub_instruction: # use sub instructions
#         # get the number of sub instructions in each sample
#         sub_num = (instruction == self.config.SPLIT_INDEX).sum(dim=1)
#         total = sub_num.sum().item()
#         max_sub_num = sub_num.max().item()
#         # construct sub_ tensors
#         sub_instructions = torch.zeros((total, self.config.SUB_PAD_SIZE)).to(device)
#         sub_lengths = torch.zeros((total,), dtype=torch.long)
#         sub_embedded = torch.zeros((total, self.config.SUB_PAD_SIZE, self.embedding_layer.embedding_dim)).to(device)
#         sub_output = torch.zeros((batch_size, max_sub_num, self.config.SUB_PAD_SIZE, self.output_size)).to(device)
#         sub_representation = torch.zeros((batch_size, max_sub_num, self.output_size)).to(device)
#         batch_idx = [[0,0]]*total

#         # encode sub instructions
#         start_index = torch.zeros((batch_size))
#         k = 0
#         for i in range(batch_size):
#             start_idx = 0
#             for j in range(sub_num[i]):
#                 end_idx = torch.argmax((instruction[i][start_idx:]==self.config.SPLIT_INDEX).int()).cpu().item()+1
#                 sub_len = end_idx
#                 end_idx += start_idx
#                 sub_instructions[k, 0:sub_len] = instruction[i][start_idx:end_idx]
#                 sub_embedded[k, 0:sub_len] = embedded[i][start_idx:end_idx]
#                 sub_lengths[k] = sub_len
#                 start_idx = end_idx
#                 batch_idx[k] = [i,j]
#                 k += 1
#         sub_packed_seq = nn.utils.rnn.pack_padded_sequence(
#             sub_embedded, sub_lengths, batch_first=True, enforce_sorted=False
#         )
#         sub_output_, sub_final_state = self.encoder_rnn_local(sub_packed_seq)

#         # post process
#         if self.config.rnn_type == "LSTM":
#             sub_final_state = sub_final_state[0]
#         sub_output_ = nn.utils.rnn.pad_packed_sequence(sub_output_, batch_first=True, total_length=self.config.SUB_PAD_SIZE)[0]
#         for k in range(total):
#             sub_representation[batch_idx[k][0],batch_idx[k][1]] = self._aggregate(sub_output_[k, 0:sub_lengths[k].item()])
#             sub_output[batch_idx[k][0],batch_idx[k][1]] = sub_output_[k]
#         sub_final_state = sub_final_state.squeeze(0)
#         output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
#         final_state = final_state.squeeze(0)
#         # output: (batch_size, seq_len, output_size)
#         # sub_output: (batch_size, max_sub_num, sub_seq_len, output_size)
#         # sub_representation: (batch_size, max_sub_num, output_size)
#         # sub_num: (batch_size)
#         # sub_lengths: (total_size)
#         if self.final_state_only:
#             return final_state, sub_final_state, sub_representation, sub_num, sub_lengths
#         else:
#             return output.permute(0, 2, 1), sub_output, sub_representation, sub_num, sub_lengths
#     else:
#         if self.final_state_only:
#             return final_state.squeeze(0)
#         else:
#             return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
#                 0
#             ].permute(0, 2, 1) # the permutation is for Conv1d
