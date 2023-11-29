import copy
from collections import OrderedDict
from typing import Any, List, Optional, Union, Dict, Callable

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from typing.io import IO
from src.evals.liefcoco import LiefCocoEvaluator
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.evaluations import PostProcess
from easydict import EasyDict as eDict
from src import utils
import random
py_logger = utils.get_logger(__name__)


class NRECModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.model = SimpleDenseNet(hparams=self.hparams)
        self.build_model()
        self.weight_init()  # TODO source code in magic location

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = self.hparams.model.loss_func
        if "losses_weight" in self.hparams.model:
            self.set_losses_weight()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_acc = Accuracy()
        # self.val_acc = Accuracy()
        # self.test_acc = Accuracy()
        self.evaluator = LiefCocoEvaluator()

        # for logging best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    # def forward(self, x: torch.Tensor):
    #     return self.model(x)

    def set_losses_weight(self):
        self.losses_weight = self.hparams.model.losses_weight
        layers = self.hparams.grd_decoder_layers
        key = ["loss_ce", "loss_bbox", "loss_contrastive_align", "loss_giou"]
        aux_weight_dict = {}
        for i in range(layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in self.losses_weight.items() if k in key})
        self.losses_weight.update(aux_weight_dict)

    def build_model(self):  # TODO
        self.visual_encoder = self.hparams.model.task_encoders.visual_encoder
        self.query_embed = self.hparams.model.task_encoders.query_embed
        self.input_proj = self.hparams.model.task_encoders.input_proj

        self.token_pos_eb = self.hparams.model.task_encoders.token_pos_eb
        self.tokenizer = self.hparams.model.task_encoders.tokenizer
        self.text_encoder = self.hparams.model.task_encoders.text_encoder
        self.text_feature_resizer = self.hparams.model.task_encoders.text_feature_resizer
        self.word_embeddings = self.text_encoder.embeddings.word_embeddings

        self.concat_encoder = self.hparams.model.task_encoders.concat_encoder
        self.cross_encoder = self.hparams.model.task_encoders.cross_encoder

        self.correction_decoder_obj = self.hparams.model.task_decoders.correction_decoder
        self.rationale_decoder_obj = self.hparams.model.task_decoders.rationale_decoder
        self.grounding_encoder = self.hparams.model.task_decoders.grounding.encoder

        self.grounding_decoder = self.hparams.model.task_decoders.grounding.decoder

        self.class_embed = self.hparams.model.task_headers.grounding.class_embed
        self.bbox_embed = self.hparams.model.task_headers.grounding.bbox_embed
        self.contrastive_align_projection_image = self.hparams.model.task_headers.grounding.contrastive_align_projection_image
        self.contrastive_align_projection_text = self.hparams.model.task_headers.grounding.contrastive_align_projection_text

        self.post_processor = PostProcess(self.tokenizer)

    # def step(self, batch: Any):
    #     x, y = batch
    #     logits = self.forward(x)
    #     loss = self.criterion(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        images, raw_sents, target_sents, rationales, targets = self.prepare_train_data(batch)
        encoder_result = self.encoder_process(images, raw_sents, target_sents)
        decoder_result = self.decoder_process(encoder_result, rationales)
        output = self.header_process(encoder_result, decoder_result)

        loss_output = self.criterion(output, targets, batch["positive_map_raw"])
        loss = sum(loss_output[k] * self.losses_weight[k] for k in loss_output.keys() if k in self.losses_weight)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}

    def visual_process(self, images):
        feature, pos = self.visual_encoder(images)
        memory_img, padding_mask_img = feature[-1].decompose()

        pos_grd = self.query_embed.weight
        batch_size = memory_img.shape[0]

        memory_img = self.input_proj(memory_img).flatten(2).permute(2, 0, 1)
        pos_img = pos[-1].flatten(2).permute(2, 0, 1)
        pos_grd = pos_grd.unsqueeze(1).repeat(1, batch_size, 1)
        padding_mask_img = padding_mask_img.flatten(1)
        memory_grd = torch.zeros_like(pos_grd, device=self.device)

        output = {"feature": memory_img,
                  "position": pos_img,
                  "padding_mask_img": padding_mask_img,
                  "pos_grd": pos_grd,  # grounding query
                  "memory_grd": memory_grd  # grounding feature
                  }

        return output

    def encoder_process(self, images, raw_sents, target_sents=None):
        # 1. visual encoder: this function is to extract visual features
        visual_rst = self.visual_process(images)

        # 2. text encoder : infer  raw sentence  and correct sentence ,and obtains the corresponding features
        raw_sents_info = self.inference_sentence(raw_sents)
        target_sents_info = self.inference_sentence(target_sents) if target_sents else {}

        # 3. concatenated feature ((multimodal fusion))
        concat_rst = self.multimodal_fusion(img_info=visual_rst,
                                            sentence_info=raw_sents_info)
        # 4. logical interaction
        logical_inter_rst = self.logical_interaction(concat_rst, raw_sents_info)

        output = {"visual": visual_rst,
                  "raw_sentence": raw_sents_info,
                  "target_sentence": target_sents_info,
                  "concatenated_feature": concat_rst,
                  "logical_interaction": logical_inter_rst}

        return output

    def decoder_process(self, encoder_result, rationales):
        raw_sent = encoder_result["raw_sentence"]
        target_sent = encoder_result["target_sentence"]
        logical_inter = encoder_result["logical_interaction"]
        concat_info = encoder_result["concatenated_feature"]
        visual_info = encoder_result["visual"]

        # 4. correction decoder
        correction_decoder_rst = self.correction_decoder(raw_sent, target_sent, logical_inter)

        # 5. rational decoder
        rationale_decoder_rst = self.rationale_decoder(rationales, logical_inter, raw_sent, concat_info)

        # 6. grounding decoder
        grd_rst = self.grounding_process(visual_info, logical_inter, raw_sent)

        output = {"correction_decoder": correction_decoder_rst,
                  "rationale_decoder": rationale_decoder_rst,
                  "grounding": grd_rst
                  }

        return output

    def header_process(self, encoder_result, decoder_result):
        raw_sent = encoder_result["raw_sentence"]
        target_sent = encoder_result["target_sentence"]
        logical_inter = encoder_result["logical_interaction"]
        rat_decoder_rst = decoder_result["rationale_decoder"]
        grd_decoder_rst = decoder_result["grounding"]
        cor_decoder_rst = decoder_result['correction_decoder']

        grd_output = self.grounding_header(grd_decoder_rst, raw_sent)
        rst = self.correction_and_rationale_header(raw_sent, target_sent, logical_inter, rat_decoder_rst)

        output = grd_output
        caption_cls_info = [
            {"correction_tokens_prob": cor_decoder_rst['cross_token_prob'],
             "correction_tokenized": target_sent['token'], "focus": rst.get("target_sent_focus")},
            {"rationale_tokens_prob": rat_decoder_rst['tokens_prob'],
             "rationale_tokenized": rat_decoder_rst['rationales_token'], "focus": rst.get("ration_focus")},
            {"change_tokens_prob": cor_decoder_rst['token_change_prob'],
             "change_labels": rst.get("change_labels")}
        ]
        output.update({"caption_cls_info": caption_cls_info})
        output.update(rst)
        return output

    def grounding_header(self, grounding, raw_sentence):
        raw_sent_token = raw_sentence["token"]
        memory_cs2 = grounding["concatenated_feature"]  # multimodal fusion(grounding encoder) result
        grd_rst = grounding["grounding_result"]  # grounding decoder result

        # grounding header output
        obj_cls = self.class_embed(grd_rst)
        obj_coord = self.bbox_embed(grd_rst).sigmoid()

        grd_output = {
            "object_class": obj_cls,
            "object_bbox": obj_coord
        }

        # aux output
        align_img = self.contrastive_align_projection_image(grd_rst)
        align_text = self.contrastive_align_projection_text(memory_cs2)

        proj_queries = F.normalize(align_img, p=2, dim=-1)
        proj_tokens = F.normalize(align_text.transpose(0, 1), p=2, dim=-1)

        aux_outputs = []
        for cls, bbox, query in zip(obj_cls[:-1], obj_coord[:-1], proj_queries[:-1]):
            aux_outputs.append({"pred_logits": cls,
                                "pred_boxes": bbox,
                                "proj_queries": query,
                                "proj_tokens": proj_tokens,
                                "tokenized": raw_sent_token
                                })

        output = {"aux_outputs": aux_outputs, "grounding_output": grd_output}

        # TODO adapt loss function
        output["pred_logits"] = obj_cls[-1]
        output["pred_boxes"] = obj_coord[-1]
        output["proj_queries"] = proj_queries[-1]
        output["proj_tokens"] = proj_tokens
        output["tokenized"] = raw_sent_token

        return output

    def validation_grounding_header(self, grounding, raw_sentence):
        raw_sent_token = raw_sentence["token"]
        memory_cs2 = grounding["concatenated_feature"]  # multimodal fusion(grounding encoder) result
        grd_rst = grounding["grounding_result"]  # grounding decoder result

        # grounding header output
        obj_cls = self.class_embed(grd_rst)
        obj_coord = self.bbox_embed(grd_rst).sigmoid()

        align_img = self.contrastive_align_projection_image(grd_rst)
        align_text = self.contrastive_align_projection_text(memory_cs2)

        proj_queries = F.normalize(align_img, p=2, dim=-1)
        proj_tokens = F.normalize(align_text.transpose(0, 1), p=2, dim=-1)

        # TODO adapt validation
        output = {}
        output["pred_logits"] = obj_cls[-1]
        output["pred_boxes"] = obj_coord[-1]
        output["proj_queries"] = proj_queries[-1]
        output["proj_tokens"] = proj_tokens
        output["tokenized"] = raw_sent_token

        return output

    def correction_and_rationale_header(self, raw_sentence, target_sentence, logical_interaction,
                                        rationale_decoder_rst):

        tokenized_rs, memory_rs, pmask_rs = raw_sentence["token"], raw_sentence["feature"], raw_sentence["padding_mask"]
        tokenized_ts, memory_ts, pmask_ts = target_sentence["token"], target_sentence["feature"], target_sentence[
            "padding_mask"]
        memory_cor_attn = logical_interaction["correction_attn"]
        tokenized_xs = rationale_decoder_rst["rationales_token"]

        feature_rawinfo = memory_rs * ~pmask_rs.t().unsqueeze(-1)
        feature_tarinfo = memory_ts * ~pmask_ts.t().unsqueeze(-1)
        feature_corinfo = memory_cor_attn * ~pmask_rs.t().unsqueeze(-1)

        caption_dist_info = {"feature_rawinfo": feature_rawinfo,
                             "feature_tarinfo": feature_tarinfo,
                             "feature_corinfo": feature_corinfo}

        ts_fos = []
        xs_fos = []
        change_labels = []
        for tokens_r, tokens_t, tokens_x in zip(tokenized_rs["input_ids"], tokenized_ts["input_ids"],
                                                tokenized_xs["input_ids"]):
            tokens_r_ = tokens_r.tolist()
            tokens_t_ = tokens_t.tolist()
            tokens_x_ = tokens_x.tolist()
            ts_fo, xs_fo = self.get_focus(tokens_r_, tokens_t_, tokens_x_)
            change_labels.append(self.get_change_label(tokens_r_, tokens_t_))
            ts_fos.append(ts_fo)
            xs_fos.append(xs_fo)

        change_labels = torch.stack(change_labels).t().to(self.device)
        output = {"change_labels": change_labels,
                  "target_sent_focus": ts_fos,
                  "ration_focus": xs_fos}
        output.update({"caption_dist_info": caption_dist_info})

        return output

    def inference_sentence(self, sentence, padding="longest", return_tensors="pt"):
        sentence_token = self.tokenizer.batch_encode_plus(sentence,
                                                          padding=padding,
                                                          return_tensors=return_tensors).to(
            self.device)
        encoder_text = self.text_encoder(**sentence_token)
        sentence_feature = encoder_text.last_hidden_state.transpose(0, 1)
        sentence_pmask = sentence_token.attention_mask.ne(1).bool()
        sentence_feature = self.text_feature_resizer(sentence_feature)

        output = {"token": sentence_token,
                  "feature": sentence_feature,
                  "padding_mask": sentence_pmask
                  }

        return output
        # return sentence_token, sentence_feature, sentence_pmask

    def prepare_train_data(self, batch):
        images = batch["images"]
        infos = batch["infos"]
        raw_sents = [info["raw_sent"] for info in infos]
        target_sents = [info["cor_sent"] for info in infos]
        rationales = [info["rationale"] for info in infos]

        require_keys = ["positive_map_raw", "positive_map_cor", "orig_size", "size", "boxes", "tokens_positive_raw"]
        targets = [{k: v for k, v in info.items() if k in require_keys} for info in infos]
        for target in targets:
            target["tokens_positive"] = target.pop("tokens_positive_raw")

        return images, raw_sents, target_sents, rationales, targets

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        images, raw_sents, target_sents, rationales, targets = self.prepare_validation_data(batch)
        encoder_result = self.encoder_process(images, raw_sents)
        decoder_result = self.validation_decoder_process(encoder_result, rationales)

        output = self.validation_grounding_header(decoder_result["grounding"],
                                                  encoder_result["raw_sentence"])

        output["caption_correction"] = decoder_result['correction_decoder']['caption_correction']
        output["caption_rationale"] = decoder_result['rationale_decoder']['caption_rationale']

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.post_processor(output, orig_target_sizes, batch["infos"])
        return results

    def test_step(self, batch: Any, batch_idx: int):
        images, raw_sents, target_sents, rationales, targets = self.prepare_validation_data(batch)
        encoder_result = self.encoder_process(images, raw_sents)
        decoder_result = self.validation_decoder_process(encoder_result, rationales)

        output = self.validation_grounding_header(decoder_result["grounding"],
                                                  encoder_result["raw_sentence"])

        output["caption_correction"] = decoder_result['correction_decoder']['caption_correction']
        output["caption_rationale"] = decoder_result['rationale_decoder']['caption_rationale']

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.post_processor(output, orig_target_sizes, batch["infos"])
        return results

    def validation_decoder_process(self, encoder_result, rationales):
        raw_sent = encoder_result["raw_sentence"]
        logical_inter = encoder_result["logical_interaction"]
        concat_info = encoder_result["concatenated_feature"]
        visual_info = encoder_result["visual"]

        # 4. correction decoder
        # correction_decoder_rst = self.correction_decoder(raw_sent, target_sent, logical_inter)
        correction_decoder_rst = self.validation_correction_decoder(raw_sent, logical_inter)

        # 5. rational decoder
        # rationale_decoder_rst = self.rationale_decoder(rationales, logical_inter, raw_sent, concat_info)
        rationale_decoder_rst = self.validation_rationale_decoder(logical_inter, raw_sent, concat_info)

        # 6. grounding decoder
        grd_rst = self.grounding_process(visual_info, logical_inter, raw_sent)

        output = {"correction_decoder": correction_decoder_rst,
                  "rationale_decoder": rationale_decoder_rst,
                  "grounding": grd_rst
                  }

        return output

    def prepare_validation_data(self, batch):
        images = batch["images"]
        infos = batch["infos"]
        raw_sents = [info["raw_sent"] for info in infos]
        target_sents = [info["cor_sent_list"] for info in infos]
        rationales = [info["rationale_list"] for info in infos]

        require_keys = ["positive_map_raw", "positive_map_cor_first", "orig_size", "size", "boxes"]
        targets = [{k: v for k, v in info.items() if k in require_keys} for info in infos]
        return images, raw_sents, target_sents, rationales, targets

        # loss, preds, targets = self.step(batch)
        #
        # # log val metrics
        # acc = self.val_acc(preds, targets)
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        #
        # return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # print(f"{len(outputs)} * {len(outputs[0])}")
        self.evaluator.update(outputs)
        liefcoco_res = self.evaluator.summarize()
        for key in liefcoco_res.keys():
            # self.log(f"val/{key}", liefcoco_res[key], on_epoch=True, prog_bar=True)
            py_logger.info(f"{key}: scores: {liefcoco_res[key]}")

        # self.log("val/grounding", liefcoco_res['liefcoco']['grounding_score'][1], on_epoch=True, prog_bar=True)

        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    # def test_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)
    #
    #     # log test metrics
    #     acc = self.test_acc(preds, targets)
    #     self.log("test/loss", loss, on_step=False, on_epoch=True)
    #     self.log("test/acc", acc, on_step=False, on_epoch=True)
    #
    #     return {"loss": loss, "preds": preds, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    # def on_epoch_end(self):
    #     # reset metrics at the end of every epoch
    #     self.train_acc.reset()
    #     self.test_acc.reset()
    #     self.val_acc.reset()

    def configure_optimizers(self):  # TODO
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        # params = {}
        # for kk, m in self.model.items():
        #     for k, v in m.items():
        #         params[k] = v
        #         break
        # model_parameters = filter(lambda p: p.requires_grad, params["visual_encoder"].parameters())
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        model_parameters = self.set_optimizer_params()
        optimizer = torch.optim.Adam(params=model_parameters,
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)

        from .nrec_scheduler import NRECScheduler
        scheduler_kwargs = self.hparams.nrec_scheduler
        scheduler_kwargs.optimizer = optimizer
        scheduler_kwargs.epochs = self.trainer.max_epochs
        scheduler_kwargs.step_size = self.trainer.max_epochs * len(self.trainer.datamodule.data_train)
        scheduler = NRECScheduler(**scheduler_kwargs)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
                }
        # return torch.optim.Adam(params=model_parameters,
        #                         lr=self.hparams.lr,
        #                         weight_decay=self.hparams.weight_decay)

        # return torch.optim.Adam(
        #     params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        # )

    # def on_train_batch_start(self, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    #
    #
    # def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    #     pass

    def correction_decoder(self, raw_sent, target_sent, logical_inter):
        pmask_cs = raw_sent["padding_mask"]  # cross padding mask

        # Denoised Feauture is equal to correction signal and noisy feature
        memory_cs = logical_inter["correction_attn"] + raw_sent["feature"]

        tok_t, bs, _ = target_sent["feature"].shape
        tokens_ts = target_sent["token"].data["input_ids"]  # target sentence token
        pmask_ts = target_sent["padding_mask"]

        pos_ts = self.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(self.device), bs)
        emb_ts = self.word_embeddings(tokens_ts).transpose(0, 1)
        pos_rs = logical_inter["position_embed"]

        token_prob, change_prob = self.correction_decoder_obj(memory_cs=memory_cs,
                                                              pmask_cs=pmask_cs,
                                                              pos_rs=pos_rs,
                                                              emb_ts=emb_ts,
                                                              pmask_ts=pmask_ts[:, :len(pos_ts)],
                                                              pos_ts=pos_ts)
        output = {"cross_token_prob": token_prob,
                  "token_change_prob": change_prob,
                  "denoised_feauture": memory_cs}

        return output

    def validation_correction_decoder(self, raw_sent, logical_inter):
        pmask_cs = raw_sent["padding_mask"]  # cross padding mask

        # Denoised Feauture is equal to correction signal and noisy feature
        memory_cs = logical_inter["correction_attn"] + raw_sent["feature"]

        # tok_t, bs, _ = target_sent["feature"].shape
        # tokens_ts = target_sent["token"].data["input_ids"]  # target sentence token
        # pmask_ts = target_sent["padding_mask"]
        bs = pmask_cs.shape[0]
        tokens_ts = torch.zeros([bs, 1], dtype=torch.long, device=self.device)
        while sum([self.tokenizer.eos_token_id in token_ids for token_ids in tokens_ts]) < bs and tokens_ts.shape[
            1] < len(memory_cs) * 2:
            pos_ts = self.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(self.device), bs)
            emb_ts = self.word_embeddings(tokens_ts).transpose(0, 1)
            pos_rs = logical_inter["position_embed"]
            pmask_ts = torch.zeros_like(tokens_ts, dtype=torch.bool, device=self.device)

            token_prob, change_prob = self.correction_decoder_obj(memory_cs=memory_cs,
                                                                  pmask_cs=pmask_cs,
                                                                  pos_rs=pos_rs,
                                                                  emb_ts=emb_ts,
                                                                  pmask_ts=pmask_ts[:, :len(pos_ts)],
                                                                  pos_ts=pos_ts)
            tokens_ts_add = token_prob.argmax(-1).t()[:, -1:]
            tokens_ts = torch.cat([tokens_ts, tokens_ts_add], dim=1)

        output = {"caption_correction": tokens_ts,
                  # "token_change_prob": change_prob,
                  "denoised_feauture": memory_cs}

        return output

    def validation_rationale_decoder(self, logical_inter, raw_sentence, concat_info):
        memory_rat_attn, pos_rs = logical_inter["rationale_attn"], logical_inter["position_embed"]
        memory_rs, pmask_rs = raw_sentence["feature"], raw_sentence["padding_mask"]
        pmask_cat, pos_cat = concat_info["padding_mask"], concat_info["position"]

        # tokenized_xs = self.tokenizer.batch_encode_plus(rationales, padding="longest", return_tensors="pt").to(
        #     self.device)
        # tok_x = tokenized_xs.data["input_ids"].shape[1]
        # tokens_xs = tokenized_xs.data["input_ids"]
        # pos_xs = self.token_pos_eb(torch.arange(tok_x).to(self.device), len(rationales))

        bs = pmask_cat.shape[0]
        tokens_xs = torch.zeros([bs, 1], dtype=torch.long, device=self.device)

        memory_rat = torch.cat([memory_rat_attn, memory_rs], dim=0)
        pmask_rat = torch.cat([pmask_cat, pmask_rs], dim=1)
        pos_rat = torch.cat([pos_cat, pos_rs], dim=0)
        while sum([self.tokenizer.eos_token_id in token_ids for token_ids in tokens_xs]) < bs and tokens_xs.shape[
            1] < 64:
            pos_xs = self.token_pos_eb(torch.arange(tokens_xs.shape[1], device=self.device), bs)
            pmask_xs = torch.zeros_like(tokens_xs, dtype=torch.bool, device=self.device)
            emb_xs = self.word_embeddings(tokens_xs).transpose(0, 1)
            tokens_prob_xs = self.rationale_decoder_obj(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)
            tokens_xs_add = tokens_prob_xs.argmax(-1).t()[:, -1:]
            tokens_xs = torch.cat([tokens_xs, tokens_xs_add], dim=1)

        # pmask_xs = tokenized_xs.attention_mask.ne(1).bool()
        # emb_xs = self.word_embeddings(tokens_xs).transpose(0, 1)
        # tokens_prob_xs = self.rationale_decoder_obj(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)

        output = {"caption_rationale": tokens_xs}

        return output

    def rationale_decoder(self, rationales, logical_inter, raw_sentence, concat_info):
        memory_rat_attn, pos_rs = logical_inter["rationale_attn"], logical_inter["position_embed"]
        memory_rs, pmask_rs = raw_sentence["feature"], raw_sentence["padding_mask"]
        pmask_cat, pos_cat = concat_info["padding_mask"], concat_info["position"]

        tokenized_xs = self.tokenizer.batch_encode_plus(rationales, padding="longest", return_tensors="pt").to(
            self.device)
        tok_x = tokenized_xs.data["input_ids"].shape[1]
        tokens_xs = tokenized_xs.data["input_ids"]
        pos_xs = self.token_pos_eb(torch.arange(tok_x).to(self.device), len(rationales))

        memory_rat = torch.cat([memory_rat_attn, memory_rs], dim=0)
        pmask_rat = torch.cat([pmask_cat, pmask_rs], dim=1)
        pos_rat = torch.cat([pos_cat, pos_rs], dim=0)
        pmask_xs = tokenized_xs.attention_mask.ne(1).bool()
        emb_xs = self.word_embeddings(tokens_xs).transpose(0, 1)
        tokens_prob_xs = self.rationale_decoder_obj(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)

        output = {"tokens_prob": tokens_prob_xs,
                  "rationales_token": tokenized_xs}

        return output

    def grounding_process(self, visual_info, logical_inter, raw_sent):
        memory_img = visual_info["feature"]
        pmask_img = visual_info["padding_mask_img"]
        pos_img = visual_info["position"]
        memory_grd = visual_info["memory_grd"]
        pos_grd = visual_info["pos_grd"]

        memory_cs = logical_inter["correction_attn"] + raw_sent["feature"]
        pmask_rs = raw_sent["padding_mask"]

        # 6. grounding decoder
        #    1)multimodal fusion
        memory_cot = torch.cat([memory_img, memory_cs], dim=0)
        pmask_cot = torch.cat([pmask_img, pmask_rs], dim=1)
        pos_cot = torch.cat([pos_img, torch.zeros_like(memory_cs)], dim=0)
        memory_cot = self.grounding_encoder(memory_cot, src_key_padding_mask=pmask_cot, pos=pos_cot)
        memory_cs2 = memory_cot[-len(memory_cs):]

        # 2) grounding decoder
        # grd decoder
        grounding_result = self.grounding_decoder(memory_grd, memory_cot, memory_cs2,
                                                  memory_key_padding_mask=pmask_cot,
                                                  text_memory_key_padding_mask=pmask_rs, pos=pos_cot,
                                                  query_pos=pos_grd)
        grounding_result = grounding_result.transpose(1, 2)

        ouput = {"grounding_result": grounding_result,
                 "concatenated_feature": memory_cs2}
        return ouput

    def multimodal_fusion(self, img_info, sentence_info):
        memory_cat = torch.cat([img_info.get("feature"), sentence_info.get("feature")], dim=0)
        pmask_cat = torch.cat([img_info.get("padding_mask_img"), sentence_info.get("padding_mask")], dim=1)
        position_cat = torch.cat([img_info.get("position"), torch.zeros_like(sentence_info.get("feature"))], dim=0)
        memory_cat = self.concat_encoder(src=memory_cat,
                                         src_key_padding_mask=pmask_cat,
                                         pos=position_cat)
        output = {"feature": memory_cat,
                  "position": position_cat,
                  "padding_mask": pmask_cat,
                  }
        return output

    def logical_interaction(self, concat_info, sentence_info):
        tok_r, bs, _ = sentence_info.get("feature").shape
        pos_rs = self.token_pos_eb(torch.arange(tok_r).to(self.device), bs)
        rationale_attn, correction_attn = self.cross_encoder(memory_cat=concat_info["feature"],
                                                             pos_cat=concat_info["position"],
                                                             pmask_cat=concat_info["padding_mask"],
                                                             memory_rs=sentence_info["feature"],
                                                             pmask_rs=sentence_info["padding_mask"],
                                                             pos_rs=pos_rs)
        output = {"rationale_attn": rationale_attn,
                  "correction_attn": correction_attn,
                  "position_embed": pos_rs}

        return output

    def get_focus(self, tokens_r_, tokens_t_, tokens_x_):
        min_len = min(len(tokens_r_), len(tokens_t_))
        for ks in range(min_len):
            if tokens_r_[ks] != tokens_t_[ks]:
                break
        for ke in range(min_len):
            if tokens_r_[-1 - ke] != tokens_t_[-1 - ke]:
                break
        focus_ts = [ks, min(len(tokens_t_) - 1, max(len(tokens_t_) - ke, ks + 1))]
        tokens_fr_ = tokens_r_[focus_ts[0]: min(len(tokens_r_) - 1, max(len(tokens_r_) - ke, ks + 1))]
        tokens_ft_ = tokens_t_[focus_ts[0]: focus_ts[1]]
        tokens_s = [tok for tok in tokens_fr_ + tokens_ft_ if tok not in self.tokenizer.all_special_ids]

        ts_fo = [k for k in list(range(focus_ts[0], focus_ts[1])) if tokens_t_[k] not in self.tokenizer.all_special_ids]
        xs_fo = [k for k in range(len(tokens_x_)) if tokens_x_[k] in tokens_s]

        return torch.tensor(ts_fo, device=self.device), torch.tensor(xs_fo, device=self.device)

    def get_change_label(self, tokens_r_, tokens_t_):
        min_len = min(len(tokens_r_), len(tokens_t_))
        for ks in range(min_len):
            if tokens_r_[ks] != tokens_t_[ks]:
                break
        for ke in range(min_len):
            if tokens_r_[-1 - ke] != tokens_t_[-1 - ke]:
                break
        focus_ts = [ks, min(len(tokens_t_) - 1, max(len(tokens_t_) - ke, ks))]
        labels = torch.tensor([0 for _ in range(len(tokens_t_))])
        labels[focus_ts[0]: focus_ts[1]] = 1
        return labels

    def set_optimizer_params(self):
        optimizer_params = self.hparams.optimizer_params

        param_dicts = []
        # base_lrs = []  # LRscheduler has done
        for k, v in optimizer_params.items():
            param_dicts.append(
                {"params": [p for n, p in self.named_parameters() if k in n and p.requires_grad], "lr": v})
            # base_lrs.append(v)

        other_params = []
        for n, p in self.named_parameters():
            if not sum([opk in n for opk in optimizer_params.keys()]) and p.requires_grad:
                other_params.append(p)

        param_dicts.append({"params": other_params, "lr": self.hparams.lr})
        # base_lrs.append(self.hparams.lr)

        return param_dicts
        # return param_dicts, base_lrs

    def weight_init(self):
        from torch import nn
        [nn.init.xavier_uniform_(p) for p in self.parameters() if p.dim() > 1]
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    # TODO
    # @classmethod
    # def load_from_checkpoint(
    #         cls,
    #         checkpoint_path: Union[str, IO],
    #         map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    #         hparams_file: Optional[str] = None,
    #         strict: bool = True,
    #         **kwargs,
    # ):
    #     old_model = torch.load(checkpoint_path, map_location="cpu")
    #
    #     ck = cls.my_on_load_checkpoint(old_model)
    #     return cls.load_state_dict(ck["state_dict"],False)
    #
    @classmethod
    def ck_match_model(cls, checkpoint: Dict[str, Any]) -> None:
        old_2_new = {
            "cat_encoder": "concat_encoder",
            "grd_decoder": "grounding_decoder",
            "cot_encoder": "grounding_encoder",
            "crs_encoder": "cross_encoder",
            "cor_decoder": "correction_decoder_obj",
            "rat_decoder": "rationale_decoder_obj",
            "resizer": "text_feature_resizer",
            "cat_encoder": "concat_encoder",
        }
        delete_key = [""]
        model_state = checkpoint['state_dict']  # TODO
        model_state_dict = OrderedDict()

        for k, v in model_state.items():
            old_name = k.split(".")[0]
            if old_name in old_2_new.keys():
                model_state_dict[k.replace(old_name, old_2_new[old_name])] = copy.deepcopy(v)

        model_state.update(model_state_dict)

        # TODO
        # checkpoint["global_step"] = self.trainer.fit_loop.global_step
        # checkpoint["epoch"] = self.trainer.fit_loop.current_epoch

        return model_state

    # @classmethod
    # def load_state_dict(cls, state_dict: 'OrderedDict[str, Tensor]',
    #                     strict: bool = True):
    #     return super().load_state_dict(state_dict, strict)

    # @classmethod
    def load_from_checkpoint(
            self,
            checkpoint_path: Union[str, IO],
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            hparams_file: Optional[str] = None,
            strict: bool = True,
            **kwargs,
    ):
        ck = torch.load(checkpoint_path)
        ck = self.ck_match_model(checkpoint=ck)
        self.load_state_dict(state_dict=ck, strict=False)
        return self
        # return super().load_from_checkpoint(checkpoint_path=checkpoint_path)
