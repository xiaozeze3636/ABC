import torch
import numpy as np
from libcity.executor.contra_mlm_executor import ContrastiveMLMExecutor
from libcity.model import loss
from tqdm import tqdm
import torch.nn.functional as F


class ContrastiveSplitMLMExecutor(ContrastiveMLMExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        # 初始化损失权重
        self.mlm_ratio = config.get('mlm_ratio', 1.0)
        self.contra_ratio = config.get('contra_ratio', 1.0)
        self.save_ratio = 0.1  # 保存10%的数据（可调整）
        self.X_raw_list = []
        self.z1_list = []
        self.labels_list = []

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)

        self.model = self.model.train()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            contra_view1, contra_view2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
                X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
            # X/targets/target_masks: (batch_size, padded_length, feat_dim)
            # padding_masks/1/2: (batch_size, padded_length)
            # batch_temporal_mat/1/2: (batch_size, padded_length, padded_length)
            contra_view1 = contra_view1.to(self.device)
            contra_view2 = contra_view2.to(self.device)
            padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
            padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
            batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
            batch_temporal_mat2 = batch_temporal_mat2.to(self.device)
            X = X.to(self.device)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(self.device)

            graph_dict = self.graph_dict

            z1, z2, predictions_l, contrastive_loss = self.model(contra_view1=contra_view1, contra_view2=contra_view2,
                                                                 argument_methods1=self.data_argument1,
                                                                 argument_methods2=self.data_argument2,
                                                                 masked_input=X, padding_masks=padding_masks,
                                                                 batch_temporal_mat=batch_temporal_mat,
                                                                 padding_masks1=padding_masks1,
                                                                 batch_temporal_mat1=batch_temporal_mat1,
                                                                 padding_masks2=padding_masks2,
                                                                 batch_temporal_mat2=batch_temporal_mat2,
                                                                 graph_dict=graph_dict)
            # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
            with torch.no_grad():
                # 获取标签（假设targets[..., 0]是类别标签）
                labels = targets[..., 0].cpu().numpy()  # (B, T)

                # 展平批次和时间步维度
                B, T = X.shape[0], X.shape[1]
                X_flat = X.cpu().numpy().reshape(B * T, -1)  # (B*T, input_dim)
                z1_flat = z1.cpu().numpy().reshape(B * T, -1)  # (B*T, embedding_dim)
                labels_flat = labels.reshape(-1)  # (B*T, )

                # 按比例随机采样
                num_samples = int(len(X_flat) * self.save_ratio)
                if num_samples > 0:
                    indices = np.random.choice(len(X_flat), num_samples, replace=False)
                    self.X_raw_list.append(X_flat[indices])
                    self.z1_list.append(z1_flat[indices])
                    self.labels_list.append(labels_flat[indices])

            targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
            mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
            # mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)
            # print(self.mlm_ratio) 0.6
            # print(self.contra_ratio) 0.4
            # byol_loss = F.mse_loss(z1, z2.detach())
            # print(11111111111111111111)
            # print(mean_loss_l)
            # print(byol_loss)
            mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * contrastive_loss
            # print(mean_loss)
            # print(22222222222222222)
            # if self.test_align_uniform or self.train_align_uniform:
            #     align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
            #     if self.train_align_uniform:
            #         mean_loss += align_uniform_loss

            if self.l2_reg is not None:
                total_loss = mean_loss + self.l2_reg * loss.l2_reg_loss(self.model)
            else:
                total_loss = mean_loss
            # print(total_loss)
            # print(3)
            total_loss = total_loss / self.grad_accmu_steps
            # print(total_loss)
            # print(3)
            batches_seen += 1

            # with torch.autograd.detect_anomaly():
            total_loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                "MLM loss": mean_loss_l.item(),
                # "BYOL loss": byol_loss.item(),
                "contrastive_loss": contrastive_loss.item(),

            }
            # if self.test_align_uniform or self.train_align_uniform:
            #     post_fix['align_loss'] = align_loss
            #     post_fix['uniform_loss'] = uniform_loss
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
        total_correct_l = total_correct_l / total_active_elements_l * 100.0
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                          .format(self.exp_id, epoch_idx, epoch_loss, total_correct_l))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        self._writer.add_scalar('Train loc acc', total_correct_l, epoch_idx)
        return epoch_loss, total_correct_l

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                contra_view1, contra_view2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
                    X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
                # X/targets/target_masks: (batch_size, padded_length, feat_dim)
                # padding_masks/1/2: (batch_size, padded_length)
                # batch_temporal_mat/1/2: (batch_size, padded_length, padded_length)
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
                batch_temporal_mat2 = batch_temporal_mat2.to(self.device)
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)

                z1, z2, predictions_l, contrastive_loss = self.model(contra_view1=contra_view1,
                                                                     contra_view2=contra_view2,
                                                                     argument_methods1=self.data_argument1,
                                                                     argument_methods2=self.data_argument2,
                                                                     masked_input=X, padding_masks=padding_masks,
                                                                     batch_temporal_mat=batch_temporal_mat,
                                                                     padding_masks1=padding_masks1,
                                                                     batch_temporal_mat1=batch_temporal_mat1,
                                                                     padding_masks2=padding_masks2,
                                                                     batch_temporal_mat2=batch_temporal_mat2,
                                                                     graph_dict=self.graph_dict)

                # === 新增代码：按比例保存数据（与_train_epoch相同） ===
                with torch.no_grad():
                    labels = targets[..., 0].cpu().numpy()  # (B, T)
                    B, T = X.shape[0], X.shape[1]
                    X_flat = X.cpu().numpy().reshape(B * T, -1)
                    z1_flat = z1.cpu().numpy().reshape(B * T, -1)
                    labels_flat = labels.reshape(-1)

                    num_samples = int(len(X_flat) * self.save_ratio)
                    if num_samples > 0:
                        indices = np.random.choice(len(X_flat), num_samples, replace=False)
                        self.X_raw_list.append(X_flat[indices])
                        self.z1_list.append(z1_flat[indices])
                        self.labels_list.append(labels_flat[indices])
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                # mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                # byol_loss = F.mse_loss(z1, z2.detach())
                mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * contrastive_loss
                # mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con

                # if self.test_align_uniform or self.train_align_uniform:
                #     align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                #     if self.train_align_uniform:
                #         mean_loss += align_uniform_loss

                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': targets_l[target_masks_l].reshape(-1, 1).squeeze(-1).cpu().numpy(),
                        # (num_active, )
                        'loc_pred': predictions_l[target_masks_l].reshape(-1, predictions_l.shape[-1]).cpu().numpy()
                        # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)

                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

                post_fix = {
                    "mode": "Train",
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                    "MLM loss": mean_loss_l.item(),
                    # "BYOL loss": byol_loss.item(),

                    "contrastive_loss": contrastive_loss.item(),
                }
                # if self.test_align_uniform or self.train_align_uniform:
                #     post_fix['align_loss'] = align_loss
                #     post_fix['uniform_loss'] = uniform_loss
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
            total_correct_l = total_correct_l / total_active_elements_l * 100.0
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                              .format(mode, self.exp_id, epoch_idx, epoch_loss, total_correct_l))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)
            self._writer.add_scalar('{} loc acc'.format(mode), total_correct_l, epoch_idx)

            if mode == 'Test':
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss, total_correct_l

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        # 调用父类训练逻辑
        super().train(train_dataloader, eval_dataloader, test_dataloader)
        # 或自定义训练循环（根据实际代码结构）

        # === 新增代码：保存数据到文件 ===
        if len(self.X_raw_list) > 0:
            import os
            import numpy as np

            # 合并所有批次的数据
            X_raw_all = np.concatenate(self.X_raw_list, axis=0)
            z1_all = np.concatenate(self.z1_list, axis=0)
            labels_all = np.concatenate(self.labels_list, axis=0)

            # 创建目录并保存
            os.makedirs("saved_embeddings", exist_ok=True)
            np.save("saved_embeddings/X_raw.npy", X_raw_all)
            np.save("saved_embeddings/z1_embeddings.npy", z1_all)
            np.save("saved_embeddings/labels.npy", labels_all)
            self._logger.info("Embeddings saved for visualization.")
