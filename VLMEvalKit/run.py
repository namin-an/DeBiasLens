import json
import os
import subprocess
from functools import partial


# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )


from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


# Make WORLD_SIZE invisible when build models
def build_model_from_config(cfg, model_name, use_vllm=False):
    import vlmeval.api
    import vlmeval.vlm
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    config = cp.deepcopy(cfg[model_name])
    if use_vllm:
        config['use_vllm'] = use_vllm
    if 'class' not in config:
        return supported_VLM[model_name](**config)
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        model = getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        model = getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')

    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=int, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')
    parser.add_argument('--alpha', type=float, default=0.6)

    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'
    alpha = args.alpha

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from vlmeval.config import api_models as supported_APIs
            from vlmeval.api import GPT4V
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, f'{model_name}_{str(alpha*100)}', eval_id)
        pred_root_meta = osp.join(args.work_dir, f'{model_name}_{str(alpha*100)}')
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, f'{model_name}_{str(alpha*100)}'), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm)

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            # try:
            result_file_base = f'{model_name}_{dataset_name}.xlsx'

            if use_config:
                if WORLD_SIZE > 1:
                    if RANK == 0:
                        dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    dist.barrier()
                dataset = build_dataset_from_config(cfg['data'], dataset_name)
                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                    continue
            else:
                dataset_kwargs = {}
                if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                    dataset_kwargs['model'] = model_name

                # If distributed, first build the dataset on the main process for doing preparation works
                if WORLD_SIZE > 1:
                    if RANK == 0:
                        dataset = build_dataset(dataset_name, **dataset_kwargs)
                    dist.barrier()

                dataset = build_dataset(dataset_name, **dataset_kwargs)
                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                    continue

            # Handling Multi-Turn Dataset
            if dataset.TYPE == 'MT':
                result_file_base = result_file_base.replace('.xlsx', '.tsv')

            result_file = osp.join(pred_root, result_file_base)
            # Reuse the previous prediction file if exists
            if RANK == 0 and len(prev_pred_roots):
                prepare_reuse_files(
                    pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                    dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                )

            if WORLD_SIZE > 1:
                dist.barrier()

            if model is None:
                model = model_name  # which is only a name

            if "_attach" in model_name:
                import sys
                sys.path.append('/workspace/cvml_user/namin/bias_vlm/sae-for-vlm')
                from dictionary_learning.trainers import MatroyshkaBatchTopKSAE

                if 'llava' in model_name:
                    from models.llava import Llava

                    model = Llava()

                    expansion_factor = 8
                    data_type = 'fairface'
                    clip_type = 'vit-large-patch14-336'
                    layer = 23
                    epoch = 100000

                    sae_path = f"/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/checkpoints_dir/matroyshka_batch_top_k_20_x{expansion_factor}/random_k_2/{data_type}_train_activations_clip-{clip_type}_{layer}_post_mlp_residual_matroyshka_batch_top_k_20_x{expansion_factor}/trainer_0/checkpoints/ae_{epoch}.pt"
                    sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()
                    neurons_to_fix = {250:0, 66:0} # gender neuron
                    model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix, pre_zero=False, alpha=alpha)
                    
                elif 'inter' in model_name:
                    from models.internlm import InternVL2
                    model = InternVL2()

                    expansion_factor = 8
                    data_type = 'fairface'
                    clip_type = 'InternViT-300M-448px'
                    layer = 23
                    epoch = 100000

                    sae_path = f"/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/checkpoints_dir/matroyshka_batch_top_k_20_x{expansion_factor}/random_k_2/{data_type}_train_activations_{clip_type}_{layer}_post_mlp_residual_matroyshka_batch_top_k_20_x{expansion_factor}/trainer_0/checkpoints/ae_{epoch}.pt"
                    sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()
                    # neurons_to_fix =  {651:0, 192:0} #{386:0, 196:0}  # gender neuron
                    data = [
                    # {0: [2342, 7090, 1389, 612, 4494, 784, 6924, 5509, 3344, 2534, 1679, 3837, 1833, 1639, 2026, 5553, 3504, 7320, 2578, 2622, 2956, 5837, 2525, 6428, 7820, 4716, 7532, 3076, 4005, 4534, 2819, 5985, 6245, 6653, 3377, 6774, 4351, 3091, 6086, 6760, 5140, 4329, 5978, 2480, 7006, 3072, 6787, 2538, 3626, 4036, 6313, 2329, 4074, 6697, 7816, 6009, 7489, 4777, 4458, 4122, 7503, 7949, 8144, 6720, 5764, 6852, 6574, 5363, 7284, 3776, 1969, 7769, 5574, 4772, 4463, 3560, 5112, 5441, 4760, 6861, 6157, 4050, 2328, 4385, 3749, 3949, 5116, 4492, 4865, 6114, 6101, 6811, 6334, 5063, 5796, 7654, 7079, 6389, 5743, 8162, 5074, 6153, 6436, 4811, 7000, 4503, 3867, 4641, 3097, 6040, 4415, 6277, 5669, 4812, 4915, 7863, 6942, 4114], 1: [601, 1628, 1223, 5221, 2524, 7535, 3408, 6353, 4365, 4294, 1103, 6007, 6017, 4715, 7940, 6315, 3393, 6173, 1865, 2959, 6888, 7870, 3825, 3986, 7642, 7034, 7889, 5729, 7479, 2535, 3621, 3028, 4448, 7452, 6513, 6264, 1742, 3850, 5761, 1951, 6340, 4395, 6370, 7864, 669, 3988, 7833, 5769, 3750, 7893, 6439, 4014, 5942, 3300, 7610, 4796, 5566, 7365, 4775, 2483, 6901, 5677, 5892, 7655, 3758, 7124, 5988, 6549, 3858, 3628, 8136, 8083, 6828, 4154, 3954, 8010, 5386, 7662, 3100, 5041, 5345, 6244, 4416, 6420, 6255, 4470, 7702, 5520, 7028, 3946, 5946, 4867, 3496, 6564, 5314, 4373, 6247, 3636, 3743, 8091, 6557, 7765, 6054, 4996, 4917, 3615, 6589, 7285, 6283, 3810, 4506, 5333, 7907, 1761, 4979, 5161, 4562, 2459, 3801, 6584, 6233, 6079, 4766, 6154, 7020, 6628, 5393, 5096, 4134, 6952, 4088, 4258, 4333, 6417, 5963]},

                    # {0: [7235, 7404, 1479, 658, 281, 7806, 1395, 3312, 3341, 6027, 6242, 1901, 1565, 3191, 3037, 2159, 623, 2571, 2490, 1093, 3023, 4460, 2658, 3981, 4759, 4213, 643, 2821, 4284, 1255, 1997, 4765, 6056, 4042, 7757, 6858, 3823, 3350, 2453, 2070, 3557, 4087, 724, 5466, 2076, 3279, 1604, 7784, 796, 7085, 4357, 6372, 6397, 3964, 8059, 2011, 1559, 4948, 5823, 520, 6882, 5584, 6490, 6333, 1652, 6792, 5902, 4913, 1397, 4604, 7202, 3009, 5084, 2837, 3252, 2104, 1678, 7632, 3989, 4001, 2847, 4414, 5862, 2082, 6805, 6187, 2765, 3473, 2284, 4081, 4733, 1798, 7957, 2041, 1794, 7927, 5909, 7993, 7934, 5562, 2408, 4315, 6649, 8081, 4511, 5934, 2827, 4232, 6815, 4341, 6280, 5602, 1562, 7172, 2270, 8191, 6630, 6707, 5692, 2692, 4669, 7022, 6551, 5525, 4251, 6396, 1872, 5370, 1682, 8146, 6087, 4692, 4371, 4943, 3422, 2281, 5478, 5655, 5451, 4047, 4803, 5724, 6703, 6770, 4583, 4891, 4635, 8044, 7055, 5174, 3707, 3304, 5172, 6554, 6424, 3779, 6526, 6503, 5348, 3574, 3035, 5678, 6543, 6466, 2363, 7711, 5285, 6367, 4837, 4175, 4059, 7207, 6392, 7411, 4301, 5983, 1819, 2529, 2795, 5356, 5549, 3917, 4851, 3804, 1914, 6926, 5899, 6248, 4788, 7967, 7848, 4768, 2030, 6057, 5100, 3683, 7548, 7842, 4489, 5334, 8085, 6055, 6691, 6778, 5984, 4847, 6098, 6048, 7015, 6884, 4048, 6700, 7086, 2126, 4535, 4577, 7935, 5780, 4354, 2450, 2632, 5950, 5814, 4565, 4417, 5704, 5401, 7912, 4255, 6969, 2556, 7623, 3740, 3925, 2561, 7377, 6650, 6677, 6868, 8178, 5982, 5407, 5904, 7615, 6995, 7229, 7169, 5767, 7496, 5565, 3179, 4719, 3840, 4168, 6245, 4740, 6500, 5350, 6385, 5556, 4752, 7319, 6613, 7856, 4029, 5362, 7270, 6922, 3612, 6796, 7164, 6236, 6516, 5976, 7506, 7372, 7597, 7700, 6978, 6771, 6676, 3163, 6086, 6218, 5813, 5465, 7778, 4352, 4121, 6874, 1656, 2175, 6108, 1838, 7922, 1851, 7273, 5854, 5833, 7918, 4162, 5015, 3881, 4869, 4012, 7956, 5182, 4505, 4559, 4808, 6966, 3452, 5755, 5194, 5731, 6436, 4503, 6040], 1: [7570, 8142, 4886, 3516, 2887, 778, 5111, 2229, 1513, 2033, 3671, 2222, 5469, 5149, 7633, 4410, 5144, 2374, 7722, 5771, 7057, 3678, 5004, 2995, 4538, 2315, 6810, 3722, 6143, 1038, 5929, 5076, 5728, 7814, 6257, 1643, 1668, 4332, 7535, 3540, 5038, 7749, 1721, 5304, 1615, 4582, 4662, 3770, 3217, 3589, 3915, 2713, 3190, 6202, 7644, 6717, 5596, 5198, 3411, 6797, 2359, 4407, 4844, 5630, 5535, 6841, 6880, 714, 6665, 5817, 6997, 5666, 4140, 6147, 4425, 7846, 6007, 5312, 7406, 6814, 804, 3601, 7264, 3206, 3018, 5428, 2137, 1176, 5083, 3900, 6476, 5460, 7072, 4239, 7800, 4013, 1829, 5866, 1574, 7707, 5748, 6565, 8180, 7278, 6017, 5424, 6203, 3510, 7031, 7640, 6947, 2144, 7422, 4720, 6807, 2553, 4950, 4513, 3487, 7583, 7421, 7786, 6542, 1984, 6735, 6173, 3388, 2681, 6463, 5331, 3869, 8026, 3219, 6446, 3913, 6604, 4637, 3624, 4592, 5893, 2793, 5658, 1907, 4380, 7323, 4260, 6937, 6546, 5514, 4554, 4152, 4010, 6508, 5483, 5209, 8122, 5855, 7310, 5233, 5050, 3808, 5828, 4998, 3649, 7467, 5002, 7288, 4401, 5851, 5949, 1400, 4095, 4992, 6817, 1674, 1378, 5142, 2318, 4795, 7049, 5432, 5664, 4419, 2289, 4292, 3271, 4281, 3379, 4190, 3044, 7244, 7479, 6801, 7147, 1723, 3652, 1973, 5513, 7501, 7877, 5868, 6575, 7741, 8189, 4784, 6580, 5302, 7919, 6528, 3444, 3523, 4619, 3201, 3506, 6090, 7683, 2417, 5129, 7617, 7433, 5947, 4995, 5758, 6194, 4220, 7453, 3604, 8113, 5733, 5418, 6846, 3105, 4734, 8151, 5672, 6898, 4573, 3980, 4898, 2152, 5080, 5541, 3939, 1911, 3716, 3635, 4149, 4859, 5750, 2662, 6749, 6582, 4225, 4230, 5262, 8061, 3992, 6296, 5614, 6437, 7864, 7860, 7313, 5818, 1972, 5745, 5214, 4027, 2369, 7303, 8051, 7833, 7545, 5421, 3741, 5958, 4033, 3122, 6013, 5018, 3899, 1804, 7095, 8036, 4939, 7462, 4630, 6175, 6640, 4375, 5279, 5256, 5211, 4904, 6688, 5361, 6709, 4227, 7135, 5501, 4906, 3893, 4706, 7286, 2274, 6144, 8158, 5892, 7661, 4928, 8126, 7478, 5547, 5212, 5563, 4725, 4778, 6845, 6459, 3661, 6816]},

                    # {0: [282, 1593, 281, 1462, 1165, 8005, 3827, 7440, 817, 3103, 5675, 985, 997, 6497, 2583, 1648, 5459, 1441, 5043, 2131, 8172, 2362, 3836, 974, 5078, 6306, 7387, 3662, 6970, 2854, 4954, 3766, 2982, 5974, 3812, 1695, 3046, 6042, 5945, 4374, 1120, 6763, 7251, 980, 831, 7725, 2457, 4213, 2435, 7828, 1734, 1030, 4209, 841, 5512, 2299, 2869, 7713, 5968, 1998, 7168, 2702, 2913, 1376, 4025, 3381, 5987, 1931, 591, 4128, 8070, 7565, 1697, 3121, 7824, 4575, 7335, 2079, 7955, 4106, 7675, 1693, 2052, 3256, 6625, 5228, 2242, 3019, 6921, 7757, 6300, 7195, 6297, 4797, 4348, 6149, 5889, 3065, 824, 6608, 7978, 6862, 7575, 6002, 1775, 5627, 2636, 4461, 4241, 6930, 718, 2056, 4234, 3897, 1858, 3885, 6050, 2504, 7731, 6839, 7813, 3871, 1778, 2533, 2844, 8043, 6648, 5167, 4986, 4880, 4497, 8118, 8018, 4002, 4907, 3942, 3700, 2735, 1898, 5684, 2160, 8133, 2576, 3385, 3482, 4713, 1729, 3902, 2581, 7301, 3759, 3975, 4991, 7346, 2614, 694, 6371, 6980, 7260, 4530, 2405, 7063, 7593, 6948, 5199, 2513, 7962, 5607, 7109, 1560, 6927, 7005, 3778, 5616, 5641, 4228, 1576, 1862, 3564, 3857, 7663, 6938, 7887, 7475, 3078, 4547, 1028, 5737, 7584, 3698, 4561, 4030, 8041, 7758, 4948, 5913, 7231, 3531, 2367, 6790, 4976, 5415, 5384, 757, 8148, 5887, 6201, 6419, 7671, 4581, 1774, 3047, 5927, 5010, 6733, 8086, 5911, 3924, 3753, 6384, 2043, 5749, 7607, 5253, 3265, 7185, 5604, 5712, 2754, 7547, 7567, 5819, 5528, 7574, 4604, 7851, 7143, 7204, 3033, 2802, 5437, 6663, 2098, 5603, 5099, 3666, 3572, 7632, 5032, 2775, 5160, 4855, 5223, 1566, 7872, 7327, 3530, 2700, 7790, 4889, 5973, 7697, 7219, 6563, 6399, 5652, 4367, 6453, 8104, 3415, 6074, 7272, 2157, 2230, 5217, 4830, 6286, 7941, 1836, 2978, 5335, 4564, 6311, 4649, 2257, 6431, 5128, 4703, 2830, 4104, 6363, 5296, 7539, 8094, 4640, 7250, 3006, 8048, 7942, 1856, 6718, 7271, 7957, 3998, 5450, 6567, 2093, 7732, 4276, 8110, 3637, 6844, 5220, 5909, 3659, 8087, 2149, 4882, 6909, 5656, 5649, 5831, 6078, 7658, 7972, 5647, 1635, 6407, 6571, 4665, 6655, 8181, 7409, 4107, 6006, 6291, 4681, 6393, 6965, 1978, 6217, 6687, 7119, 6474, 4818, 5635, 6707, 5382, 7022, 6724, 4598, 6759, 6293, 3525, 6934, 4892, 6051, 7181, 3653, 5895, 5720, 4215, 5655, 6391, 8116, 3958, 7666, 4268, 5526, 4197, 6770, 5507, 4678, 7582, 5061, 2268, 5590, 4806, 7711, 4301, 4275, 7687, 5414, 3962, 7804, 4510, 2372, 3898, 5814, 7001, 5401, 2556, 7615, 6922, 6218, 5813, 7449, 4352, 6874, 2175, 1838, 5833, 5731, 6436, 6040], 1: [2570, 3773, 566, 3343, 2271, 2747, 8027, 2006, 1755, 4612, 7177, 4625, 2613, 3232, 2347, 6413, 7192, 1034, 1711, 793, 7347, 4886, 2665, 1158, 6032, 2194, 1315, 2212, 3856, 1813, 3260, 4443, 527, 1830, 2990, 5410, 7013, 7849, 2633, 5111, 7685, 6929, 603, 7773, 2763, 2628, 5706, 1501, 6818, 2652, 7861, 1513, 1624, 6489, 5681, 2033, 512, 1751, 2222, 6418, 5625, 1660, 7674, 5469, 6599, 5521, 3948, 7835, 6510, 8067, 5000, 2850, 5144, 5164, 1512, 1191, 1700, 849, 1601, 3538, 4690, 561, 3498, 2488, 5097, 3295, 2200, 3239, 6366, 5555, 6450, 2113, 7057, 4520, 6030, 4170, 6651, 3821, 5280, 3440, 5104, 5474, 6085, 6454, 2133, 5282, 4966, 7766, 7522, 7869, 6736, 4793, 3846, 1156, 1996, 7538, 2669, 1764, 7525, 4989, 4946, 7868, 1133, 4959, 6475, 6810, 7196, 5923, 6136, 3722, 5691, 5405, 4466, 4143, 3172, 3235, 5550, 1718, 1038, 5035, 4514, 2114, 1791, 4139, 7814, 6257, 7991, 3119, 2066, 6024, 4774, 5624, 7966, 7578, 5765, 6165, 4313, 7812, 5070, 3611, 7535, 2565, 7266, 6750, 7394, 6073, 6111, 5068, 3364, 7412, 3258, 7928, 6338, 2128, 4062, 5304, 710, 1902, 2648, 6344, 4582, 688, 5788, 3770, 5088, 3217, 5336, 5353, 2508, 6517, 2998, 2213, 3894, 3589, 7603, 3190, 2353, 4949, 7159, 6414, 6238, 6717, 1987, 2496, 5198, 7417, 3712, 6797, 8092, 2814, 5297, 7796, 6285, 2926, 3613, 4844, 7282, 3502, 4422, 6592, 5847, 4167, 5564, 5237, 4198, 3711, 6841, 4481, 6606, 4895, 4150, 6997, 5778, 6290, 6547, 3639, 4425, 7846, 7657, 6670, 4685, 5803, 5374, 4325, 7579, 6876, 2148, 7132, 5081, 5829, 6620, 6295, 6814, 3394, 7609, 3206, 4951, 7133, 5428, 3689, 3562, 1816, 8131, 4587, 4671, 3900, 3586, 6317, 5460, 7117, 4145, 7127, 5270, 7997, 5640, 3227, 6096, 7974, 6900, 4776, 2216, 6172, 5751, 2953, 7053, 7332, 5147, 5166, 3928, 7278, 7759, 6612, 2584, 4366, 4764, 6017, 2671, 5424, 4964, 7550, 3855, 2788, 7330, 6117, 6947, 3692, 4092, 6064, 2882, 3849, 5998, 3629, 6807, 7065, 3844, 5141, 2553, 1800, 8119, 7455, 5340, 7269, 4487, 5092, 3845, 5874, 5559, 6020, 1753, 7958, 6957, 3177, 6052, 7583, 3971, 5932, 6590, 7405, 5427, 6897, 3799, 7391, 6177, 5920, 5990, 4820, 1631, 6199, 5278, 4260, 6343, 7782, 7080, 4747, 4554, 4646, 4429, 7933, 5483, 7551, 5209, 6511, 1855, 4597, 6022, 5828, 4135, 5849, 3878, 3649, 4593, 6390, 2125, 4095, 5661, 5047, 6627, 4080, 7088, 5142, 4418, 3645, 4419, 5179, 6092, 7220, 6813, 4900, 3658, 7581, 7326, 6528, 6737, 5192, 4721, 4591, 6194, 2547, 4112, 5224, 3595, 7076, 4214, 5786]},

                    # {0: [648, 231, 2589, 320, 656, 920, 3696, 1006, 935, 4299, 538, 281, 576, 1251, 2013, 1450, 840, 532, 1458, 4392, 8049, 3103, 2366, 7827, 2989, 1125, 5048, 3216, 7843, 7819, 1396, 2855, 2659, 1805, 7636, 7951, 1360, 5546, 7963, 7705, 2620, 3521, 2256, 1287, 6917, 2278, 1274, 6352, 1630, 2885, 7716, 2124, 3305, 1696, 3223, 2791, 2177, 1012, 1949, 1999, 1894, 747, 4498, 7044, 2663, 1229, 3240, 2106, 2005, 5444, 562, 2939, 621, 6740, 3376, 3426, 5059, 2736, 4118, 1511, 1611, 1246, 7840, 6307, 3319, 3428, 736, 3552, 6713, 7139, 6373, 7691, 770, 6945, 2833, 1909, 4379, 1698, 7134, 6150, 5295, 7454, 6369, 1647, 2465, 3905, 2742, 2380, 4499, 3220, 1908, 3966, 2798, 2111, 2667, 1892, 5694, 1522, 1986, 6654, 2384, 2976, 3051, 2458, 5397, 6492, 7818, 2979, 1961, 2876, 6883, 2911, 3815, 2810, 6066, 5524, 4213, 7828, 2102, 2653, 6999, 3099, 3468, 1734, 1590, 2825, 6331, 4965, 3479, 5650, 3994, 4000, 2797, 1623, 2299, 4101, 7513, 2664, 926, 3165, 6902, 3794, 3178, 3873, 6430, 7210, 6282, 4557, 6541, 7193, 3883, 2969, 7777, 4616, 2285, 5006, 3107, 6323, 6326, 780, 3451, 2336, 7426, 7824, 4575, 4125, 7367, 5809, 1808, 1792, 7675, 1870, 1693, 1852, 5705, 7469, 6597, 2037, 7191, 5785, 5943, 3282, 6272, 7874, 6679, 3598, 3558, 7205, 6067, 5463, 4304, 7902, 5244, 6728, 3877, 7728, 4647, 6820, 1879, 7112, 5319, 6766, 5417, 1642, 4182, 5267, 7976, 3995, 5177, 3669, 6309, 7805, 5125, 4477, 6461, 5369, 857, 7151, 7389, 5639, 2621, 2866, 4905, 2215, 2499, 6415, 2896, 1942, 5258, 3321, 3374, 6742, 5977, 8057, 6954, 6025, 2365, 7456, 6835, 3871, 7989, 6992, 2844, 5305, 3461, 5644, 7811, 6132, 7084, 4634, 4712, 5579, 3493, 6987, 2423, 2900, 4011, 3891, 2796, 6482, 3587, 5329, 3318, 5381, 1983, 6780, 3420, 2263, 3806, 7656, 4501, 4602, 4108, 2160, 4672, 3237, 5721, 5082, 4779, 6624, 7572, 4714, 6347, 1603, 2405, 7295, 2430, 7694, 6462, 5029, 5960, 6200, 5944, 6783, 8149, 3564, 2706, 3703, 5508, 6499, 8124, 4971, 6416, 6865, 6060, 1028, 6949, 3698, 3585, 5017, 4704, 3908, 6678, 2645, 8041, 7358, 7246, 2766, 4976, 3952, 2395, 5927, 7447, 2048, 5712, 2754, 7429, 4846, 7214, 5528, 4156, 6985, 6944, 7851, 6213, 4226, 2098, 2775, 4261, 6091, 6179, 6563, 5652, 4367, 5795, 6074, 7952, 6812, 2257, 4703, 2830, 5445, 4104, 2320, 7539, 3006, 3998, 3637, 6407, 7409, 6393, 6707, 5382, 6293, 3525, 5402, 6391, 4678, 7001], 1: [1269, 1440, 3391, 1563, 1499, 1085, 521, 3576, 3568, 741, 698, 3193, 826, 1465, 1261, 566, 7155, 1069, 3752, 3343, 523, 553, 8027, 1423, 775, 7239, 4940, 2243, 6683, 1755, 894, 7719, 6400, 8177, 1401, 3400, 660, 4625, 3309, 3249, 3384, 2012, 764, 4071, 6413, 6657, 984, 5156, 3074, 772, 3965, 1034, 4178, 2851, 1711, 6195, 370, 1958, 1283, 5435, 4386, 4884, 2409, 6032, 6849, 5375, 4600, 2212, 3856, 1813, 4443, 6345, 7152, 2990, 7648, 7909, 7314, 7013, 7849, 2633, 1536, 5111, 7685, 6933, 2901, 6929, 1854, 3754, 4252, 7998, 4550, 4006, 7770, 1151, 7861, 1421, 4207, 7140, 6837, 7183, 4008, 7891, 5307, 5138, 577, 2096, 4576, 4519, 868, 4742, 2033, 3186, 2219, 2893, 3049, 6418, 6863, 5044, 4411, 2432, 6804, 1262, 5469, 2314, 6599, 2695, 4126, 2008, 6510, 3979, 5000, 2850, 5103, 8046, 5164, 7638, 1700, 5052, 2849, 1187, 5419, 6498, 2832, 3336, 3040, 5097, 4826, 6647, 2361, 635, 6075, 2200, 2464, 8077, 2156, 3014, 8128, 2424, 7057, 2325, 1975, 7201, 4096, 1688, 3676, 4605, 6769, 6632, 2804, 4854, 6406, 6788, 6701, 6069, 6705, 6388, 1279, 3474, 4769, 4926, 2481, 1620, 5619, 5085, 3561, 3281, 7396, 4347, 7847, 1451, 4793, 2674, 5448, 2402, 905, 3888, 4171, 4350, 7839, 3233, 6485, 4201, 7030, 6714, 3596, 3244, 4476, 8186, 8185, 7196, 3945, 6136, 5959, 4453, 3317, 8179, 4143, 3834, 3235, 3298, 5398, 4514, 6610, 3387, 4139, 7991, 2947, 7424, 6996, 5071, 6206, 4120, 4039, 5867, 7858, 4322, 8082, 6103, 2872, 7966, 7578, 6165, 2709, 4264, 4313, 1713, 7353, 5070, 2300, 1977, 6073, 6111, 7412, 3694, 7630, 3583, 5653, 6637, 6827, 2846, 5304, 1471, 4750, 878, 5739, 6913, 7290, 2508, 4390, 2526, 7595, 6120, 7603, 6956, 3884, 3959, 6989, 7470, 6405, 2340, 7888, 6312, 5999, 7977, 2843, 2920, 1550, 4310, 5281, 5168, 7296, 3829, 3755, 7237, 4858, 7517, 7306, 4481, 3695, 4148, 2324, 5760, 6290, 5841, 1579, 4718, 4873, 5829, 7645, 4916, 5479, 8003, 4951, 6142, 5628, 4113, 2881, 4389, 3650, 3882, 7997, 7634, 4208, 3865, 5055, 4744, 4092, 4896, 5042, 4229, 3274, 7080]},

                    # {0: [152, 175, 436, 320, 3696, 1943, 565, 954, 2418, 1177, 6907, 3990, 6842, 1476, 2875, 1267, 1899, 642, 702, 2221, 538, 1080, 610, 281, 771, 853, 3410, 5034, 580, 5242, 1009, 1607, 7983, 927, 1847, 2349, 3543, 4479, 2279, 1241, 3316, 650, 3327, 917, 6350, 5583, 1354, 1490, 2165, 1242, 2989, 1910, 1464, 2909, 706, 2543, 2936, 1844, 2555, 6016, 6932, 993, 3477, 607, 6168, 5673, 1431, 1543, 4165, 4194, 2975, 4785, 1275, 2444, 1291, 8093, 2620, 2562, 4758, 1247, 3004, 6018, 755, 2406, 4205, 2643, 4041, 4085, 3026, 7815, 3677, 773, 1161, 2874, 7113, 6873, 8139, 5494, 3524, 3464, 7986, 2495, 3101, 3228, 2273, 2922, 7472, 5567, 1810, 4359, 4183, 7350, 3416, 4297, 1971, 2938, 2307, 3261, 2644, 3240, 2897, 2949, 2848, 2062, 589, 4633, 7682, 2398, 8182, 621, 5136, 3038, 6850, 2183, 2800, 2542, 7495, 6536, 7305, 3417, 2569, 3923, 6409, 6731, 1572, 6445, 2401, 2390, 4200, 3195, 3592, 7014, 7924, 6036, 2441, 1840, 4502, 6725, 4937, 5367, 6830, 5790, 7558, 1599, 7402, 3466, 2338, 4300, 4881, 6150, 6754, 2670, 3934, 4546, 7211, 6303, 7948, 4499, 5582, 7703, 2731, 1849, 4541, 6587, 7791, 5598, 7830, 3051, 3324, 7987, 1670, 6851, 3007, 6912, 5784, 3815, 3904, 5342, 6432, 7647, 7880, 7695, 4045, 3684, 1569, 3851, 6039, 6059, 1613, 3099, 6386, 5446, 3468, 7068, 6808, 6672, 7457, 3994, 5189, 6319, 5937, 3726, 5321, 4849, 4131, 4402, 3165, 5836, 7388, 7210, 7115, 5310, 7777, 3107, 6323, 3451, 7602, 1792, 4188, 2880, 2018, 5341, 6411, 2132, 7205, 1768, 3877, 3620, 5412, 6820, 2952, 2859, 3003, 5639, 2896, 3445, 3321, 4610, 3088, 2925, 7881, 7096, 6224, 8063, 4011, 3891, 6596, 1983, 6780, 2263, 3237, 2405, 7295, 5885, 7460, 6775, 5029, 6783, 2706, 6499, 7206, 6416, 6865, 1028, 5017, 3908, 6678, 2395, 7447, 2048, 7851, 6179, 6393, 5382, 6391], 1: [452, 1770, 2155, 1602, 779, 1189, 3391, 709, 1070, 1563, 6169, 1209, 1155, 2895, 3109, 1730, 1406, 2189, 2774, 1499, 3550, 1222, 2228, 814, 1341, 7886, 2206, 3368, 698, 1258, 1250, 1303, 3532, 3176, 1658, 990, 2744, 3125, 7410, 2053, 523, 5487, 3991, 5049, 6382, 3688, 6568, 1195, 7689, 2243, 1064, 2977, 6683, 859, 5879, 2069, 3069, 5723, 6823, 2109, 7321, 8177, 6176, 1401, 3243, 3337, 1864, 660, 4625, 6832, 2910, 6116, 4071, 818, 6920, 5156, 3848, 3259, 3275, 3140, 1412, 4178, 2071, 7635, 2980, 3170, 2482, 2776, 1905, 6271, 4756, 1663, 2554, 5394, 3056, 7755, 4884, 2224, 6849, 6833, 5375, 3082, 2544, 2932, 3856, 1813, 7879, 7961, 1936, 2147, 1758, 5178, 5930, 2352, 2677, 7909, 5372, 1815, 1900, 2616, 7944, 4362, 7685, 6933, 3866, 7488, 2489, 3754, 7998, 2040, 5569, 4550, 7502, 5229, 1421, 7331, 6616, 3602, 7050, 2286, 5072, 6914, 2789, 3841, 5856, 2028, 4161, 3854, 5250, 7381, 7668, 3547, 2065, 3777, 6010, 5914, 7425, 6804, 5469, 5311, 6599, 2255, 4908, 2164, 3267, 5103, 7007, 5726, 3352, 8012, 3710, 5137, 5052, 2443, 2849, 7459, 4819, 6498, 3610, 6821, 7795, 3580, 3785, 1782, 2421, 2474, 2427, 4545, 8040, 6406, 6788, 7081, 1627, 6943, 2509, 5679, 3474, 6163, 5701, 4123, 2873, 4746, 5954, 6029, 6232, 2756, 2478, 3682, 5287, 4387, 5991, 6714, 4674, 8186, 7234, 5670, 2078, 3748, 8184, 3123, 3198, 1669, 5259, 6165, 7419, 3694, 4833, 7905, 5001, 3803, 6689]},

                    # {0: [769, 1472, 109, 540, 2546, 6259, 613, 802, 536, 1295, 2790, 1216, 1500, 2682, 2739, 1203, 1509, 2039, 1875, 619, 1041, 758, 633, 4522, 1299, 1003, 3196, 7821, 2612, 3126, 6225, 1691, 1115, 957, 927, 2279, 3386, 3111, 981, 5685, 3291, 2154, 1188, 3414, 2127, 3399, 3154, 2086, 2337, 888, 768, 3114, 700, 4295, 2745, 2265, 2493, 1139, 685, 2592, 2543, 3325, 4184, 2470, 2195, 3174, 3581, 1828, 2168, 1524, 3110, 6967, 2036, 8167, 3403, 2807, 1275, 2171, 4644, 8093, 3472, 2705, 1017, 4507, 630, 4491, 2411, 4224, 8068, 1553, 2240, 1185, 2406, 3693, 1013, 2473, 4085, 2287, 2771, 3931, 7815, 5091, 7622, 1650, 1161, 2937, 3404, 5494, 1362, 7986, 3269, 7854, 1609, 8016, 5247, 5013, 2962, 7161, 6674, 5738, 4297, 2683, 1912, 3194, 7224, 2966, 2848, 589, 5131, 1596, 8187, 4069, 5136, 4259, 4142, 4321, 7495, 3322, 3086, 3147, 4032, 3266, 3421, 1796, 6182, 1591, 6044, 6036, 4434, 7558, 5533, 1599, 3466, 4300, 1880, 5540, 4679, 6212, 3115, 1681, 1849, 5783, 4541, 3324, 2085, 6432, 6427, 6386, 3468, 5937, 2859, 3088, 4011, 7460, 2048], 1: [1134, 1770, 2155, 2593, 763, 1263, 641, 1602, 779, 1132, 513, 3354, 1170, 2715, 7370, 1007, 1070, 1563, 2004, 691, 2895, 1234, 865, 1289, 2189, 2595, 1846, 5996, 1554, 923, 1129, 3209, 814, 1365, 2100, 3148, 798, 703, 3368, 2024, 5876, 3013, 2787, 2416, 6544, 1303, 939, 2051, 1245, 3532, 1886, 6126, 696, 3149, 1063, 6364, 2986, 990, 2919, 2184, 3125, 2852, 3518, 7410, 2053, 959, 3230, 2387, 5487, 2203, 2434, 2751, 1433, 6246, 1636, 7509, 3205, 5049, 3357, 3340, 2243, 2199, 7106, 4783, 1336, 6683, 1214, 965, 1573, 2069, 2178, 2886, 3427, 5723, 1699, 1595, 5502, 7321, 3243, 6469, 3337, 1891, 4852, 2512, 3301, 4925, 5360, 6116, 2560, 1850, 2015, 1680, 4071, 2559, 1581, 2249, 727, 3533, 3055, 6920, 5395, 5023, 5151, 3275, 2071, 2150, 4756, 3094, 2799, 1790, 5394, 5406, 2609, 1694, 7787, 7650, 4789, 1737, 3135, 5352, 2376, 5753, 1947, 7772, 6422, 5919, 2489, 5699, 6752, 8001, 7915, 8112, 7163, 2262, 4805, 2835, 3500, 6621, 4344, 5469, 7052, 1915, 1941, 2625, 5026, 6848, 2858, 2185]},

                    # {0: [234, 3772, 313, 336, 468, 322, 614, 139, 1952, 3221, 655, 1300, 575, 2650, 1811, 1066, 1199, 1498, 3213, 1284, 1425, 640, 599, 1106, 1614, 24, 2075, 998, 1672, 2032, 1056, 2790, 1385, 1769, 1384, 3139, 690, 2293, 930, 744, 2682, 1922, 2536, 2039, 1799, 2635, 2046, 991, 1039, 1313, 2166, 1029, 2491, 1903, 937, 1372, 3196, 1667, 2640, 6109, 2187, 1705, 2960, 1507, 697, 2750, 2449, 1488, 812, 2439, 7262, 1390, 3207, 4543, 2154, 1188, 715, 2127, 1542, 6034, 3159, 1586, 4277, 6601, 2573, 3154, 2025, 3742, 1505, 2745, 3434, 1929, 2493, 6981, 3157, 6216, 2094, 1939, 3146, 2967, 791, 1524, 2145, 2904, 2225, 3403, 6349, 2044, 2198, 1520, 630, 6550, 2931, 6011, 3016, 4736, 6643, 2567, 1860, 3404, 1609, 1873, 2962, 1912, 2966, 2848, 5131, 4069, 3322, 3147, 1989, 1599], 1: [73, 1083, 1726, 1075, 595, 1483, 936, 1204, 1363, 1057, 1770, 2879, 2155, 2593, 659, 1190, 2531, 1806, 641, 1589, 1225, 1286, 2715, 7370, 666, 1007, 2004, 951, 2895, 4978, 901, 5805, 2189, 2595, 524, 942, 923, 1337, 3209, 2691, 615, 2845, 1073, 3287, 1001, 874, 858, 3148, 684, 5876, 3013, 2787, 1032, 2416, 1047, 6544, 1544, 1886, 4377, 2460, 3149, 2163, 3104, 3278, 560, 2919, 2095, 1000, 1712, 2053, 3680, 3230, 2413, 2387, 2000, 3160, 7559, 1573, 1463, 5736, 2259, 7321, 1541, 2577, 3527, 854, 1680, 4071, 2813, 5585]},

                    # {0: [314, 1159, 70, 3029, 210, 378, 649, 870, 280, 638, 1135, 652, 1219, 1312, 1519, 1478, 289, 3089, 583, 915, 1502, 3213, 2668, 1428, 1380, 1746, 759, 682, 1535, 1184, 1081, 1130, 1055, 3306, 1672, 2615, 1387, 956, 2536, 348, 4912, 7810, 6099, 653, 991, 1523, 1313, 2630, 760, 1029, 1157, 569, 1469, 2360, 1488, 1333, 2486, 6730, 2806, 1278, 2564, 1542, 2368, 2094, 1939, 3403, 3225], 1: [73, 132, 256, 22, 496, 1083, 1218, 886, 1075, 936, 1204, 661, 866, 516, 1310, 1169, 2518, 608, 1485, 1843, 1057, 1770, 960, 950, 573, 387, 427, 1198, 1096, 911, 641, 620, 750, 1108, 6798, 787, 493, 681, 1992, 1383, 1953, 867, 1271, 976, 3117, 973, 3355, 2895, 529, 1470, 598, 1033, 1272, 2189, 2595, 969, 1304, 952, 902, 2845, 3469, 1743, 2220, 1474, 2787, 1544, 1402, 1404, 2163, 4099, 3104, 1556, 6918]},

                    {0: [14, 291, 120, 293, 343, 1053, 1179, 1393, 1102, 1143, 1460, 1475, 1521, 1146, 664, 1016, 1666, 1046, 34, 571, 1366, 723, 1213, 2063, 978, 856, 7736, 438, 1523, 1313, 2806], 1: [192, 177, 399, 119, 1381, 506, 1528, 133, 1248, 568, 428, 1218, 1530, 882, 255, 143, 1109, 18, 1224, 45, 845, 895, 516, 810, 292, 482, 950, 1198, 1194, 943, 657, 750, 667, 1037, 945, 1097, 726, 3372, 1820, 1020]},

                    # {0: [], 1: [385, 113, 473]}
                    ]

                    flat = []
                    for d in data:
                        for key in d:
                            flat.extend(d[key])
                    neurons_to_fix = {k:-65000 for k in list(set(flat))}
                    model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix, pre_zero=False, alpha=alpha)


            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    result_file_name=result_file_base,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc,
                    use_vllm=args.use_vllm)
            elif dataset.TYPE == 'MT':
                model = infer_data_job_mt(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc,
                    ignore_failed=args.ignore,
                    use_vllm=args.use_vllm)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc,
                    ignore_failed=args.ignore,
                    use_vllm=args.use_vllm)
            
            
            # Set the judge kwargs first before evaluation or dumping

            judge_kwargs = {
                'nproc': args.api_nproc,
                'verbose': args.verbose,
                'retry': args.retry if args.retry is not None else 3,
                **(json.loads(args.judge_args) if args.judge_args else {}),
            }

            if args.retry is not None:
                judge_kwargs['retry'] = args.retry
            if args.judge is not None:
                judge_kwargs['model'] = args.judge
            else:
                print(dataset_name)
                if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
                    ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
                ):
                    if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['VisuLogic'], dataset_name):
                        judge_kwargs['model'] = 'exact_matching'
                    else:
                        judge_kwargs['model'] = 'chatgpt-0125'
                elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                    if listinstr(['LLaVABench_KO'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o-0806'
                    else:
                        judge_kwargs['model'] = 'gpt-4-turbo'
                elif listinstr(['VGRPBench'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning'], dataset_name):  # noqa: E501
                    judge_kwargs['model'] = 'gpt-4o-mini'
                elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):  # noqa: E501
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['ChartMimic'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['VDC'], dataset_name):
                    judge_kwargs['model'] = 'llama31-8b'
                elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
                    judge_kwargs['model'] = 'qwen-72b'
                elif listinstr(['MMVMBench'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4.1'
                elif listinstr(['M4Bench'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'

            if args.use_verifier:
                judge_kwargs['use_verifier'] = True
            if args.use_vllm:
                judge_kwargs['use_vllm'] = True

            if RANK == 0:
                logger.info(judge_kwargs)

            if WORLD_SIZE > 1:
                dist.barrier()

            # Only RANK 0 handles the evaluation part
            if RANK == 0:
                # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                if dataset_name in ['MMMU_TEST']:
                    result_json = MMMU_result_transfer(result_file)
                    logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                f'json file saved in {result_json}')
                    continue
                elif 'MMT-Bench_ALL' in dataset_name:
                    submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                    logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                f'submission file saved in {submission_file}')
                    continue

                # Skip the evaluation part if only infer
                if args.mode == 'infer':
                    continue

                # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                if 'MLLMGuard_DS' in dataset_name:
                    logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                    continue
                elif 'AesBench_TEST' == dataset_name:
                    logger.info(f'The results are saved in {result_file}. '
                                f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                    continue
                elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                    logger.info(f'{dataset_name} is a test split without ground-truth. '
                                'Thus only the inference part is supported for those datasets. ')
                    continue
                elif dataset_name in [
                    'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                    'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                ] and not MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                    continue

                # Setup the proxy for the evaluation
                eval_proxy = os.environ.get('EVAL_PROXY', None)
                old_proxy = os.environ.get('HTTP_PROXY', '')
                if eval_proxy is not None:
                    proxy_set(eval_proxy)

                # Perform the Evaluation
                eval_results = dataset.evaluate(result_file, **judge_kwargs)
                # Display Evaluation Results in Terminal
                if eval_results is not None:
                    assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                    logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                    logger.info('Evaluation Results:')
                    if isinstance(eval_results, dict):
                        logger.info('\n' + json.dumps(eval_results, indent=4))
                    elif isinstance(eval_results, pd.DataFrame):
                        if len(eval_results) < len(eval_results.columns):
                            eval_results = eval_results.T
                        logger.info('\n' + tabulate(eval_results))

                # Restore the proxy
                if eval_proxy is not None:
                    proxy_set(old_proxy)

                # Create the symbolic links for the prediction files
                files = os.listdir(pred_root)
                files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                for f in files:
                    cwd = os.getcwd()
                    file_addr = osp.join(cwd, pred_root, f)
                    link_addr = osp.join(cwd, pred_root_meta, f)
                    if osp.exists(link_addr) or osp.islink(link_addr):
                        os.remove(link_addr)
                    os.symlink(file_addr, link_addr)

            # except Exception as e:
            #     logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
            #                      'skipping this combination.')
            #     continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
