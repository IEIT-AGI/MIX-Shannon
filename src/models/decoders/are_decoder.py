import json
from typing import Optional, Dict, Callable, Tuple
import torch
from torch import nn
from omegaconf import DictConfig
from src.models.builder import DECODERS, MODELS, build_decoder, build_head, build_models
import hydra
import random


@MODELS.register_module()
class GroupMLP(nn.Module):

    def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
        super(GroupMLP, self).__init__()

        self.conv1 = nn.Conv1d(in_features, mid_features, 1)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

    def forward(self, a):
        N, C = a.size()
        h = self.relu(self.conv1(a.view(N, C, 1)))
        return self.conv2(self.drop(h)).view(N, -1)


@MODELS.register_module()
class MLP_ANS(nn.Module):

    def __init__(self, ans_feature_len, hidden_size, embedding_size):
        super(MLP_ANS, self).__init__()
        self.mlp = GroupMLP(
            in_features=ans_feature_len,  # fan
            mid_features=hidden_size,  # 2048
            out_features=embedding_size,  # fan
            drop=0.0,
            groups=64,  # 64
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, a, a_len=None):
        # pdb.set_trace()
        from torch.nn import functional as F
        return self.mlp(F.normalize(a, p=2))


@DECODERS.register_module()
class ReasoningDecoder1(nn.Module):

    def __init__(self, tokenizer: DictConfig, text_encoder: DictConfig, event_data: str, answer_data: str,
                 relation_data: str, hidden_size: int, event_net: DictConfig, answer_net: DictConfig,
                 relation_net: DictConfig, header: DictConfig, relation_map: str):
        super(ReasoningDecoder1, self).__init__()

        self.event_vectors = self.get_external_knowledge(text_encoder_cfg=text_encoder,
                                                         tokenizer_cfg=tokenizer,
                                                         file=event_data,
                                                         hidden_size=hidden_size)
        self.answer_vectors = self.get_external_knowledge(text_encoder_cfg=text_encoder,
                                                          tokenizer_cfg=tokenizer,
                                                          file=answer_data,
                                                          hidden_size=hidden_size)
        self.relation_vectors = self.get_external_knowledge(text_encoder_cfg=text_encoder,
                                                            tokenizer_cfg=tokenizer,
                                                            file=relation_data,
                                                            hidden_size=hidden_size)

        self.event_net = build_models(event_net)
        self.answer_net = build_models(answer_net)
        self.relation_net = build_models(relation_net)
        self.header = build_head(header)

        with open(relation_map) as f:
            self.relation_map_head_tail = json.load(f)
        self.transe_loss_flag = False  # fasle is  triple transe loss

    @staticmethod
    def get_external_knowledge(text_encoder_cfg: DictConfig, tokenizer_cfg: DictConfig, file: str, hidden_size: int):

        def load_json() -> Dict:
            import json
            with open(file) as f:
                content = json.load(f)
                return content["answer"]

        from torch.nn import functional as F
        from torch.autograd import Variable

        tokenizer_obj = hydra.utils.instantiate(tokenizer_cfg)
        text_encoder_obj = hydra.utils.instantiate(text_encoder_cfg)

        data: Dict = load_json()
        data_vectors = torch.zeros(len(data), hidden_size)
        for idx, dt in enumerate(data.keys()):
            dt_tokenized = tokenizer_obj.batch_encode_plus([dt], padding="longest", return_tensors="pt")
            dt_embed = text_encoder_obj.embeddings.word_embeddings(dt_tokenized.data["input_ids"]).transpose(0, 1)
            data_vectors[idx, :] = torch.mean(dt_embed, dim=0)

        return Variable(F.normalize(data_vectors, p=2, dim=1))

    def forward(self, batch: Dict, reason_encoder_out: Dict):
        event_fusion_feat: torch.Tensor = reason_encoder_out["event_feature"]
        answer_fusion_feat: torch.Tensor = reason_encoder_out["knowledge_feature"]
        relation_fusion_feat: torch.Tensor = reason_encoder_out["relation_feature"]

        event_embedding = self.event_net(self.event_vectors.to(event_fusion_feat.device))
        answer_embedding = self.answer_net(self.answer_vectors.to(event_fusion_feat.device))
        relation_embedding = self.answer_net(self.relation_vectors.to(event_fusion_feat.device))

        event_info = {"feature": event_fusion_feat, "embedding": event_embedding}
        answer_info = {"feature": answer_fusion_feat, "embedding": answer_embedding}
        relation_info = {"feature": relation_fusion_feat, "embedding": relation_embedding}

        output = self.header(batch, event_info, answer_info, relation_info)
        if self.training:
            if self.transe_loss_flag:
                transe_loss = self.transe_loss(output)
                # print(f"transe_loss:{transe_loss}")
                output.update({"loss_transe": transe_loss})
            else:
                triplet_transe_loss = self.triplet_transe_loss(batch, output)
                # print(f"transe_loss:{transe_loss}")
                output.update({"loss_triplet_transe": triplet_transe_loss})
        return output

    def transe_loss(self, header_out):

        def get_evnet_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.event_vectors[max_idx, :]
            return embedding

        def get_answer_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.answer_vectors[max_idx, :]
            return embedding

        def get_relation_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.relation_vectors[max_idx, :]
            return embedding

        event_embedding = get_evnet_embedding(header_out.get("predict_event"))
        answer_embedding = get_answer_embedding(header_out.get("predict_answer"))
        relation_embedding = get_relation_embedding(header_out.get("predict_relation"))

        # h+r-e
        transe_loss = abs(event_embedding + relation_embedding - answer_embedding).mean(dim=-1).sum()
        transe_loss *= 25

        # #
        # transe_loss = abs(event_embedding + relation_embedding - answer_embedding).sum(dim=-1).mean()
        # transe_loss *= 0.125

        return transe_loss.to(header_out.get("predict_event").device)

    def triplet_transe_loss(self, batch, header_out):

        def get_evnet_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.event_vectors[max_idx, :]
            return embedding

        def get_answer_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.answer_vectors[max_idx, :]
            return embedding

        def get_relation_embedding(predict):
            predict_softmax = predict.softmax(-1)
            max_idx = predict_softmax.argmax(-1)
            embedding = self.relation_vectors[max_idx, :]
            return embedding

        def get_false_answer():
            # fact_label = batch.get("fact_label")
            # fact_label_idx = [d.argmax() for d in fact_label]
            #
            answer_label = batch.get("answer_label")
            answer_label_label_idx = [d.argmax() for d in answer_label]

            relation_label = batch.get("relation_label")
            relation_label_idx = [d.argmax() for d in relation_label]

            false_answer_idx = []
            for relation_idx, answer_idx in zip(relation_label_idx, answer_label_label_idx):
                other_answer_idx = []
                for idx in self.relation_map_head_tail.keys():
                    if int(idx) != relation_idx.item():
                        other_answer_idx.extend(self.relation_map_head_tail[idx].get("tail"))
                # if relation_idx.item() in other_answer_idx: # todo too small probability
                #     other_answer_idx.remove(relation_idx.item())
                false_answer_idx.append(random.choice(other_answer_idx))

            answer_embedding = self.answer_vectors[false_answer_idx, :]
            return answer_embedding

        event_embedding = get_evnet_embedding(header_out.get("predict_event"))  # fact
        answer_embedding = get_answer_embedding(header_out.get("predict_answer"))
        relation_embedding = get_relation_embedding(header_out.get("predict_relation"))

        false_answer_embedding = get_false_answer()

        # h+r-e
        true_transe_loss = abs(event_embedding + relation_embedding - answer_embedding).mean(dim=-1).sum()
        false_transe_loss = abs(event_embedding + relation_embedding - false_answer_embedding).mean(dim=-1).sum()

        loss = 3 + true_transe_loss - false_transe_loss

        # #
        # transe_loss = abs(event_embedding + relation_embedding - answer_embedding).sum(dim=-1).mean()
        # transe_loss *= 0.125

        return loss.to(header_out.get("predict_event").device)


@DECODERS.register_module()
class ExplainingDecoder1(nn.Module):  # locating key object

    def __init__(self, num_queries: int, is_pass_query: bool, class_embed: DictConfig, bbox_embed: DictConfig,
                 query_embed: DictConfig, grounding_decoder: DictConfig, header: DictConfig):
        super(ExplainingDecoder1, self).__init__()
        self.num_queries = num_queries
        self.is_pass_query = is_pass_query

        self.class_embed = nn.Linear(**class_embed)  # todo label nums ?
        self.bbox_embed = build_models(bbox_embed)
        # self.query_embed = build_embedding(query_embed)
        self.query_embed = hydra.utils.instantiate(query_embed)

        self.grd_decoder = build_decoder(grounding_decoder)
        self.header = build_head(header)

    def forward(self, encode_info: Dict, target: Optional[Dict] = None) -> Dict:
        if self.training:
            return self.forward_train(encode_info, target)
        else:
            return self.forward_test(encode_info, target)

    def forward_predict(self, encode_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encode_info["question_mask"])
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        if self.is_pass_query:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt, query_embed = query_embed, None

        hs = self.grd_decoder(tgt=tgt,
                              memory=encode_info["grounding_encoder_feature"],
                              text_memory=encode_info["question_resize_feature"],
                              memory_key_padding_mask=encode_info["grounding_mask"],
                              text_memory_key_padding_mask=encode_info["question_mask"],
                              pos=encode_info["grounding_position"],
                              query_pos=query_embed)

        hs = hs.transpose(1, 2)
        pred_class = self.class_embed(hs)
        pred_coord = self.bbox_embed(hs).sigmoid()

        return pred_class[-1], pred_coord[-1]

    def forward_test(self, encode_info: Dict, target: Dict) -> Dict:
        pred_cls, pred_coord = self.forward_predict(encode_info)
        # output
        predicts = {"pred_logics": pred_cls, "pred_boxes": pred_coord}
        header_rst = self.header(predicts, target)

        output = predicts
        output.update(header_rst)

        return output

    def forward_train(self, encode_info: Dict, target: Dict) -> Dict:
        pred_cls, pred_coord = self.forward_predict(encode_info)
        predicts = {"pred_logics": pred_cls, "pred_boxes": pred_coord}
        header_rst = self.header(predicts, target)

        # output
        output = predicts
        output.update(header_rst)
        return output


@DECODERS.register_module()
class ReasoningDecoder(nn.Module):

    def __init__(self, tokenizer: DictConfig, text_encoder: DictConfig, event_data: str, answer_data: str,
                 hidden_size: int, event_net: DictConfig, answer_net: DictConfig, header: DictConfig):
        super(ReasoningDecoder, self).__init__()

        self.event_vectors = self.get_external_knowledge(text_encoder_cfg=text_encoder,
                                                         tokenizer_cfg=tokenizer,
                                                         file=event_data,
                                                         hidden_size=hidden_size)
        self.answer_vectors = self.get_external_knowledge(text_encoder_cfg=text_encoder,
                                                          tokenizer_cfg=tokenizer,
                                                          file=answer_data,
                                                          hidden_size=hidden_size)

        self.event_net = build_models(event_net)
        self.answer_net = build_models(answer_net)
        self.header = build_head(header)

    @staticmethod
    def get_external_knowledge(text_encoder_cfg: DictConfig, tokenizer_cfg: DictConfig, file: str, hidden_size: int):

        def load_json() -> Dict:
            import json
            with open(file) as f:
                content = json.load(f)
                return content["answer"]

        from torch.nn import functional as F
        from torch.autograd import Variable

        # tokenizer_obj = build_tokenizer(tokenizer_cfg)
        # text_encoder_obj = build_text_encoder(text_encoder_cfg)
        tokenizer_obj = hydra.utils.instantiate(tokenizer_cfg)
        text_encoder_obj = hydra.utils.instantiate(text_encoder_cfg)

        data: Dict = load_json()
        data_vectors = torch.zeros(len(data), hidden_size)
        for idx, dt in enumerate(data.keys()):
            dt_tokenized = tokenizer_obj.batch_encode_plus([dt], padding="longest", return_tensors="pt")
            dt_embed = text_encoder_obj.embeddings.word_embeddings(dt_tokenized.data["input_ids"]).transpose(0, 1)
            data_vectors[idx, :] = torch.mean(dt_embed, dim=0)

        return Variable(F.normalize(data_vectors, p=2, dim=1))

    def forward(self, batch: Dict, reason_encoder_out: Dict):
        event_fusion_feat: torch.Tensor = reason_encoder_out["event_feature"]
        answer_fusion_feat: torch.Tensor = reason_encoder_out["knowledge_feature"]

        event_embedding = self.event_net(self.event_vectors.to(event_fusion_feat.device))
        answer_embedding = self.answer_net(self.answer_vectors.to(event_fusion_feat.device))

        event_info = {"feature": event_fusion_feat, "embedding": event_embedding}
        answer_info = {"feature": answer_fusion_feat, "embedding": answer_embedding}
        return self.header(batch, event_info, answer_info)


@DECODERS.register_module()
class ExplainingDecoder(nn.Module):  # locating key object

    def __init__(self, num_queries: int, is_pass_query: bool, class_embed: DictConfig, bbox_embed: DictConfig,
                 query_embed: DictConfig, grounding_decoder: DictConfig, header: DictConfig):
        super(ExplainingDecoder, self).__init__()
        self.num_queries = num_queries
        self.is_pass_query = is_pass_query

        self.class_embed = nn.Linear(**class_embed)  # todo label nums ?
        self.bbox_embed = build_models(bbox_embed)
        # self.query_embed = build_embedding(query_embed)
        self.query_embed = hydra.utils.instantiate(query_embed)

        self.grd_decoder = build_decoder(grounding_decoder)
        self.header = build_head(header)

    def forward(self, encode_info: Dict, target: Optional[Dict] = None) -> Dict:
        if self.training:
            return self.forward_train(encode_info, target)
        else:
            return self.forward_test(encode_info)

    def forward_predict(self, encode_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encode_info["question_mask"])
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        if self.is_pass_query:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt, query_embed = query_embed, None

        hs = self.grd_decoder(tgt=tgt,
                              memory=encode_info["grounding_encoder_feature"],
                              text_memory=encode_info["question_resize_feature"],
                              memory_key_padding_mask=encode_info["grounding_mask"],
                              text_memory_key_padding_mask=encode_info["question_mask"],
                              pos=encode_info["grounding_position"],
                              query_pos=query_embed)

        hs = hs.transpose(1, 2)
        pred_class = self.class_embed(hs)
        pred_coord = self.bbox_embed(hs).sigmoid()

        return pred_class[-1], pred_coord[-1]

    def forward_test(self, encode_info: Dict) -> Dict:
        pred_cls, pred_coord = self.forward_predict(encode_info)
        # output
        output = {"pred_logics": pred_cls, "pred_boxes": pred_coord}

        return output

    def forward_train(self, encode_info: Dict, target: Dict) -> Dict:
        predicts = self.forward_test(encode_info)
        header_rst = self.header(predicts, target)

        # output
        output = predicts
        output.update(header_rst)
        return output