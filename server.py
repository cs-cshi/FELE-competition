from datetime import datetime
import os
import shutil
import unittest
import torch
import numpy as np
from homo_decision_tree.homo_decision_tree_arbiter import HomoDecisionTreeArbiter
from worker import Worker
from preprocess import get_test_data
#from preprocess import get_test_loader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tree_core.quantile_summaries import quantile_summary_factory
import shutil

from context import FederatedAveragingGrads,PytorchModel
from learning_model import FLModel
import random


class ParameterServer(HomoDecisionTreeArbiter):
    def __init__(self, feature_num, boosting_round, booster_dim, bin_num, learning_rate, max_depth, testworkdir,
                 resultdir, modeldir):
        super().__init__()
        self.projects = []
        self.workers = []
        self.workers_project = []
        self.global_bin_split_points = []
        self.feature_num = feature_num
        self.boosting_round = boosting_round
        self.booster_dim = booster_dim
        self.bin_num = bin_num
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        self.modeldir = modeldir
        self.test_data = get_test_data()
        # self.test_data_loader = get_test_loader()
        self.label_distribution = []
        self.group_id = 0
        self.group_num = 12
        self.predictions = []
        self.work_parm = []

    def treeEnsemble(self, pred_results):
        print("Start to aggregate.")
        result = []
        for i in range(0, len(pred_results)):
            valid_pred = [a for a in pred_results[i] if a != 13]
            if len(valid_pred) != 0:
                result.append(np.argmax(np.bincount(valid_pred)))
            else:
                result.append(13)
        with open(os.path.join('result', 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in result]))

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray,)):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self):
        prediction = []
        predictions = self.predictions
        for rid in range(len(predictions[0])):
            predict = []
            for i in range(len(predictions)):
                predict.append(predictions[i][rid])
            prediction.append(np.argmax(np.bincount(predict)))
        self.save_prediction(prediction)

    def ensemble(self):
        for i in range(self.group_num):
            # s = random.sample(range(79), 10)
            self.workers_project = []
            for pid in self.label_distribution[i]:
                print(self.projects[pid].count_label())
                # self.projects[pid].set_data_bin_feature(slice)
                self.workers_project.append(self.projects[pid])
            self.aggregate()
            loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=1000,
                shuffle=False,
            )
            test_data = []
            with torch.no_grad():
                for data in loader:
                    data = np.array(data)
                    test_data.extend(data)
            test_data = np.array(test_data)

            self.predictions.append(self.predict_data(test_data.tolist()))

    def build(self, workers):
        for worker in workers:
            self.projects += worker.build_projects()
        label_dict = {}
        self.label_distribution = [[] for i in range(self.group_num)]
        for i in range(len(self.projects)):
            u_id, c = self.projects[i].count_label()
            for key, value in c.items():
                if key in label_dict:
                    label_dict[key].append(i)
                else:
                    label_dict[key] = [i]
        print(label_dict)
        for key, value in label_dict.items():
            for i in range(self.group_num):
                self.label_distribution[i].append(value[i])
        print(self.label_distribution)
        self.workers = workers
        print('user number is:{}'.format(len(self.workers)))

    def get_quantile(self):
        global_quantile = self.workers_project[0].receive_quantile_info()
        for worker in self.workers_project[1:]:
            summary_list = worker.receive_quantile_info()
            for fid in range(len(global_quantile)):
                global_quantile[fid].merge(summary_list[fid])
        self.global_bin_split_points = []
        percent_value = 1.0 / self.bin_num

        percentile_rate = [i * percent_value for i in range(1, self.bin_num)]
        percentile_rate.append(1.0)
        for sum_obj in global_quantile:
            split_point = []
            for percent_rate in percentile_rate:
                s_p = sum_obj.query(percent_rate)
                if s_p not in split_point:
                    split_point.append(s_p)
            self.global_bin_split_points.append(split_point)
        print(self.global_bin_split_points)

    def get_all_histogram(self, class_idx, dep):
        left_node_histogram = self.workers_project[0].receive_local_h_info(class_idx, dep)
        for worker in self.workers_project[1:]:
            worker_loacl_h = worker.receive_local_h_info(class_idx, dep)
            for nid, node in enumerate(worker_loacl_h):
                # left_node_histogram[nid].merge_hist(worker_loacl_h[nid])
                feature_hist1 = left_node_histogram[nid].bag
                feature_hist2 = worker_loacl_h[nid].bag
                assert len(feature_hist1) == len(feature_hist2)
                for j in range(len(feature_hist1)):
                    assert len(feature_hist1[j]) == len(feature_hist2[j])
                    for k in range(len(feature_hist1[j])):
                        assert len(feature_hist1[j][k]) == 3
                        feature_hist1[j][k][0] += feature_hist2[j][k][0]
                        feature_hist1[j][k][1] += feature_hist2[j][k][1]
                        feature_hist1[j][k][2] += feature_hist2[j][k][2]

        # ??????????????????histogram
        all_histograms = self.histogram_subtraction(left_node_histogram, self.stored_histograms)
        return all_histograms

    def aggregate(self):
        print('start aggregate')
        # ??????Quantile sketch???????????????????????????
        self.get_quantile()
        # ?????????worker???????????????
        for worker in self.workers_project:
            worker.fit_init(self.global_bin_split_points)
        for epoch_idx in range(self.boosting_round):
            print('epoch:{}'.format(epoch_idx))

            # ???worker?????????booster?????????????????????
            for worker in self.workers_project:
                if epoch_idx >= 1:
                    worker.choose_valid_feature_data()
                worker.fit_booster_init()
            for class_idx in range(self.booster_dim):
                print('class:{}'.format(class_idx))
                # ?????????label??????????????????????????????
                g_sum, h_sum = 0, 0
                for worker in self.workers_project:
                    # ?????????????????????
                    worker.fit_tree_init(class_idx)
                    # ????????????worker??????epoch???class????????????g h
                    g, h = worker.receive_g_h_info(class_idx)
                    g_sum += g
                    h_sum += h
                # ????????????g h??????????????????worker
                for worker in self.workers_project:
                    worker.fit_distribute_global_g_h(class_idx, g_sum, h_sum)

                tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf
                for dep in range(tree_height):
                    print("The depth is {}".format(dep))
                    if dep + 1 == tree_height:
                        # ??????????????????
                        for worker in self.workers_project:
                            worker.fit_tree_stop(class_idx)
                        break
                    cur_layer_node_num = self.workers_project[0].receive_cur_layer_node_num_info(class_idx)
                    for worker in self.workers_project[1:]:
                        assert worker.receive_cur_layer_node_num_info(class_idx) == cur_layer_node_num

                    layer_stored_hist = {}
                    all_histograms = self.get_all_histogram(class_idx, dep)
                    # store histogram
                    for hist in all_histograms:
                        layer_stored_hist[hist.hid] = hist
                    best_splits = self.federated_find_best_split(all_histograms, parallel_partitions=10)
                    self.stored_histograms = layer_stored_hist

                    for worker in self.workers_project:
                        worker.fit_distribute_split_info(class_idx, dep, best_splits)
                for worker in self.workers_project:
                    # ??????????????????bid??????????????????
                    worker.fit_convert(class_idx)

                # update predict score
                for worker in self.workers_project:
                    worker.fit_update_y_hat(class_idx, self.learning_rate, epoch_idx)
                    worker.update_feature_importance()

    def predict_data(self, data):
        return self.workers_project[0].predict(data, self.learning_rate, self.boosting_round)

class NNParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedAveragingGrads(
            model=PytorchModel(torch=torch,
                               model_class=FLModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'),
            framework='pytorch',
        )
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.test_data = get_test_data()

        self.round_train_acc= []
        self.global_feature_mean = []
        self.global_feature_std = []
                    
    def aggr_mean_std(self, feature_mean, feature_std, worker_num):
        global_feature_mean = np.zeros(79)
        global_feature_std = np.zeros(79)

        for i in range (0,79):
            cur_mean = 0
            cur_std = 0
            for j in range (0, worker_num):
                cur_mean += feature_mean[j][i]
                cur_std += feature_std[j][i]
            global_feature_mean[i] = cur_mean/worker_num
            global_feature_std[i] = cur_std/worker_num

        self.global_feature_mean = global_feature_mean
        self.global_feature_std = global_feature_std

    def test_data_norm(self):
        for i in range (0,len(self.test_data)):
            for j in range (0,len(self.test_data[0])):
                if self.global_feature_std[j] !=0 :
                    self.test_data[i][j] = (self.test_data[i][j] - self.global_feature_mean[j])/self.global_feature_std[j]
                else:
                    self.test_data[i][j] = self.test_data[i][j] - self.global_feature_mean[j]
        
    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads): # receive grads info from worker
        self.current_round_grads.append(grads)

    def receive_worker_info(self, info): # receive worker info from worker
        self.worker_info = info

    def process_round_train_acc(self): # process the "round_train_acc" info from worker
        self.round_train_acc.append(self.worker_info["train_acc"])

    def print_round_train_acc(self):
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc: ", "%.2f%%" % (mean_round_train_acc))
        self.round_train_acc = []
        return {"mean_round_train_acc": mean_round_train_acc}

    def aggregate(self):
        self.aggr(self.current_round_grads)

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray, )):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self, model, device, test_batch_size):
        loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=test_batch_size,
            shuffle=False,
        )
        prediction = []
        with torch.no_grad():
            for data in loader:
                data = torch.unsqueeze(data, dim=1)                
                pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                prediction.extend(pred.reshape(-1).tolist())

        self.save_prediction(prediction)
