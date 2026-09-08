import copy
import tqdm
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.custom_3d import Custom3DDataset

import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadOccupancy
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
import glob
import numpy as np
from .semantic_kitti_dataset import CustomSemanticKITTIDataset

@DATASETS.register_module()
class CustomSemanticKITTILssDataset(CustomSemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, random_camera=False, cbgs=False, repeat=1, queue_length=1, load_multi_voxel=False, *args, **kwargs):
        super(CustomSemanticKITTILssDataset, self).__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.random_camera = random_camera
        self.all_camera_ids = list(self.camera_map.values())
        self.load_multi_voxel = load_multi_voxel
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        self.repeat = repeat
        self.cbgs = cbgs

        if self.repeat > 1:
            self.data_infos = self.data_infos * self.repeat
            random.shuffle(self.data_infos)

        # init class-balanced sampling
        self.data_infos = self.init_cbgs()

        self._set_group_flag()

    def prepare_cat_infos(self):
        tmp_file = 'semkitti_train_class_counts.npy'

        if not os.path.exists(tmp_file):
            class_counts_list = []
            for index in tqdm.trange(len(self)):
                info = self.data_infos[index]['voxel_path']
                assert info is not None
                target_occ = np.load(info)

                # compute the class counts
                cls_ids, cls_counts = np.unique(target_occ, return_counts=True)
                class_counts = np.zeros(self.n_classes)

                cls_ids = cls_ids.astype(np.int)
                for cls_id, cls_count in zip(cls_ids, cls_counts):
                    # ignored
                    if cls_id == 255:
                        continue

                    class_counts[cls_id] += cls_count

                class_counts_list.append(class_counts)

            # num_sample, num_class
            self.class_counts_list = np.stack(class_counts_list, axis=0)
            np.save(tmp_file, self.class_counts_list)
        else:
            self.class_counts_list = np.load(tmp_file)

    def init_cbgs(self):
        if not self.cbgs:
            return self.data_infos

        self.prepare_cat_infos()
        # remove unlabel class
        self.class_counts_list = self.class_counts_list[:, 1:]
        num_class = self.class_counts_list.shape[1]

        class_sum_counts = np.sum(self.class_counts_list, axis=0)
        sample_sum = class_sum_counts.sum()
        class_distribution = class_sum_counts / sample_sum

        # compute the balanced ratios
        frac = 1.0 / num_class
        ratios = frac / class_distribution
        ratios = np.log(1 + ratios)

        sampled_idxs_list = []
        for cls_id in range(num_class):
            # number of total points for this class
            num_class_sample_pts = class_sum_counts[cls_id] * ratios[cls_id]

            # get corresponding samples
            class_sample_valid_mask = (self.class_counts_list[:, cls_id] > 0)
            class_sample_valid_indices = class_sample_valid_mask.nonzero()[0]

            class_sample_points = self.class_counts_list[class_sample_valid_mask, cls_id]
            class_sample_prob = class_sample_points / class_sample_points.sum()
            class_sample_expectation = (class_sample_prob * class_sample_points).sum()

            # class_sample_mean = class_sample_points.mean()
            num_samples = int(num_class_sample_pts / class_sample_expectation)
            sampled_idxs = np.random.choice(class_sample_valid_indices, size=num_samples, p=class_sample_prob)
            sampled_idxs_list.extend(sampled_idxs)

        sampled_infos = [self.data_infos[i] for i in sampled_idxs_list]

        return sampled_infos

    def prepare_train_data(self, index):  # 根据当前样本索引准备单帧或多帧时序训练数据
        if self.queue_length > 1:  # 时序长度大于 1 时进入多帧数据加载模式
            queue = []  # 保存每一帧经过数据处理流水线后的结果
            # * HTCL 是一个因果的、固定滑动窗口式时序 SSC 网络，核心贡献是跨帧特征对应、对齐和融合；
            # * 它不是以持续世界状态和未来演化预测为目标的 world model，也不是严格意义上的状态化流式推理系统。
            #! 原注释代码
            # index_list = list(range(index - self.queue_length, index))  # 生成当前帧之前的候选历史帧索引
            # # random.shuffle(index_list)  # 原设计可随机打乱候选历史帧，当前已禁用
            # index_list = sorted(index_list[1:])  # 删除最早的一帧，并按索引从小到大排列剩余历史帧
            # index_list.append(index)  # 将当前帧索引添加到末尾，组成“历史帧 + 当前帧”序列
            # *=======================================================================#
            #! 修改代码：避免跨场景取历史帧，确保历史帧与当前帧属于同一序列
            index_list = list(range(max(0, index - self.queue_length + 1), index + 1))  # 生成长度不超过 queue_length 的连续候选索引，并将起始索引限制为非负数
            index_list = [i for i in index_list if self.data_infos[i]["sequence"] == self.data_infos[index]["sequence"]]  # 仅保留与当前帧属于同一序列的索引，避免跨场景取历史帧
            index_list = [index_list[0]] * (self.queue_length - len(index_list)) + index_list  # 历史帧不足时在队列头部重复该序列最早的有效帧，使队列长度固定


            for i in index_list:  # 按时间索引依次处理序列中的每一帧
                i = max(0, i)  # 将负索引截断为 0，通过重复第 0 帧补齐数据集开头的历史帧
                input_dict = self.get_data_info(i)  # 获取该帧的图像路径、标定参数和体素标签等信息
                if input_dict is None:  # 检查该帧的数据索引是否有效
                    return None  # 任意一帧无效时放弃构造当前训练样本
                self.pre_pipeline(input_dict)  # 初始化数据处理流水线需要的字段
                example = self.pipeline(input_dict)  # 读取并处理图像、体素标签、深度监督等数据
                queue.append(example)  # 将处理完成的这一帧加入时序队列
            return self.union2one(queue)  # 沿时间维堆叠多帧数据，并返回以当前帧为目标的训练样本
        else:  # 时序长度不大于 1 时使用单帧数据加载模式
            input_dict = self.get_data_info(index)  # 获取当前帧的图像路径、标定参数和体素标签等信息
            if input_dict is None:  # 检查当前帧的数据索引是否有效
                return None  # 当前帧无效时不生成训练样本
            self.pre_pipeline(input_dict)  # 初始化数据处理流水线需要的字段
            example = self.pipeline(input_dict)  # 读取并处理当前帧的训练数据
            return example  # 返回处理完成的单帧训练样本

    def prepare_test_data(self, index):
        if self.queue_length>1:
            queue = []
            index_list = list(range(index-self.queue_length, index))
            # random.shuffle(index_list)
            index_list = sorted(index_list[1:])
            index_list.append(index)
            if index<3:
                for i in range(0, len(index_list)): index_list[i]=index_list[i] if index_list[i]>0 else 0

            for i in index_list:
                i = max(0, i)
                input_dict = self.get_data_info(i)
                if input_dict is None:
                    return None
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                queue.append(example)
            return self.union2one(queue)
        else:
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return example



    def union2one(self, queue):
        imgs_list0 = [each['img_inputs'][0][0].data for each in queue]
        imgs_list1 = [each['img_inputs'][1][0].data for each in queue]
        queue[-1]['img_inputs'][0][0] = DC(torch.stack(imgs_list0), cpu_only=False, stack=True)
        queue[-1]['img_inputs'][1][0] = DC(torch.stack(imgs_list1), cpu_only=False, stack=True)


        imgs_feature0 = [torch.tensor(np.asarray(each['img_inputs'][0][-1].data)) for each in queue]
        imgs_feature1 = [torch.tensor(np.asarray(each['img_inputs'][1][-1].data)) for each in queue]
        queue[-1]['img_inputs'][0][-1] = DC(torch.stack(imgs_feature0), cpu_only=False, stack=True)
        queue[-1]['img_inputs'][1][-1] = DC(torch.stack(imgs_feature1), cpu_only=False, stack=True)

        # metas_map = {}
        # gt_occ = {}
        # points_occ = {}
        # points_uv = {}
        # for i, each in enumerate(queue):
        #     metas_map[i] = each['img_metas'].data
        #     gt_occ[i] = each['gt_occ'].data
        #     points_occ[i] = each['points_occ'].data
        #     points_uv[i] = each['points_uv'].data
        #     metas_map[i]['prev_bev_exists'] = False
        # queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        # queue[-1]['gt_occ'] = DC(gt_occ, cpu_only=True)
        # queue[-1]['points_occ'] = DC(points_occ, cpu_only=True)
        # queue[-1]['points_uv'] = DC(points_uv, cpu_only=True)

        queue = queue[-1]
        return queue



    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']
        if info is None:
            return None

        if self.load_multi_voxel:
            annos = []
            for scale in self.multi_scales:
                scale_info = info.replace('1_1', scale)
                annos.append(np.load(scale_info))

            return annos
        else:
            return np.load(info)

    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            img_filename = info['img_2_path'] ,
        )

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        for cam_type in self.camera_used:
            if self.random_camera:
                cam_type = random.choice(self.all_camera_ids)

            image_paths.append(info['img_{}_path'.format(int(cam_type))])

            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])

            cam_intrinsics.append(info['P{}'.format(int(cam_type))])

            lidar2cam_rts.append(info['T_velo_2_cam'])


        calib_info = self.read_calib_file(info['calib_path'])
        calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * self.dynamic_baseline(calib_info)
        # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        input_dict.update(
            dict(
                img_filename=image_paths,

                lidar2img=lidar2img_rts,

                cam_intrinsic=cam_intrinsics,

                lidar2cam=lidar2cam_rts,

                calib = calib
            ))

        # ground-truth in shape (256, 256, 32), XYZ order
        # TODO: how to do bird-eye-view augmentation for this?
        input_dict['gt_occ'] = self.get_ann_info(index)
        return input_dict

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    def dynamic_baseline(self, calib_info):
        P3 =np.reshape(calib_info['P3'], [3,4])
        P =np.reshape(calib_info['P2'], [3,4])
        baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
        return baseline

    def evaluate(self, results, logger=None, **kwargs):
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']

            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])

            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])

            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)

            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }

        class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou

        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)

        # add two main metrics to serve as the sort metric
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']

        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)

        return eval_results
