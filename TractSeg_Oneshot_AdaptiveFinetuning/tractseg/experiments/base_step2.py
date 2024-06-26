#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

from tractseg.data import dataset_specific_utils
from tractseg.libs.system_config import SystemConfig as C


class Config:
    """
    Settings and hyperparameters
    """

    # input data
    EXPERIMENT_TYPE = "tract_segmentation"  # tract_segmentation|endings_segmentation|dm_regression|peak_regression
    EXP_NAME = "HCP_TEST"
    EXP_MULTI_NAME = ""  # CV parent directory name; leave empty for single bundle experiment
    DATASET_FOLDER = "HCP_preproc"
    LABELS_FOLDER = "bundle_masks"
    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    # CLASSES = "All"
    NR_OF_GRADIENTS = 9
    # NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(CLASSES)[1:])
    INPUT_DIM = None  # autofilled
    DATASET = "HCP"  # HCP | HCP_32g | Schizo
    RESOLUTION = "1.25mm"  # 1.25mm|2.5mm
    # 12g90g270g | 270g_125mm_xyz | 270g_125mm_peaks | 90g_125mm_peaks | 32g_25mm_peaks | 32g_25mm_xyz
    FEATURES_FILENAME = "12g90g270g"
    # LABELS_FILENAME = ""  # autofilled
    LABELS_TYPE = "int"
    THRESHOLD = 0.5  # Binary: 0.5, Regression: 0.01

    # hyperparameters
    # MODEL = 'UNet_Pytorch'
    # MODEL = 'UNet_Pytorch_DeepSup'

    


    WARM_NUM = 1 # 5/3/1 ----the subject num used for finetune based on pretrained model
    LOAD_WEIGHTS = True


    ##-----12 novel tracts-----
    tract_num = 12
    LABELS_FILENAME = "bundle_masks_"+str(tract_num)
    CLASSES = 'Other_'+str(tract_num)
    NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(CLASSES)[1:])
    AUG_NUM = 100
    TUNE_DATA = 'HCP'  # for 'HCP'
    # DATA_PATH = '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_'+str(tract_num)+'/MtractoutYKeep'

    WEIGHTS_PATH = "/data4"
    FIX_OTHER_LAYER = False
    USE_DAUG = False

    ###----------For SpotTune-------
    MODEL = 'UNet_Pytorch_Spottune'
    PMODEL = 'ResNet'
    temperature = 0.5
    CMODEL = 'UNet_Pytorch_Cnet'


    ###----Two stage finetune----------------
    #---Warmup stageI-------
    WEIGHTS_PATH = "/data6/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x30/best_weights_ep188.npz"
    FIX_OTHER_LAYER = True
    # # ##********need to change******
    USE_DAUG = False # Baseline
    # USE_DAUG = True  # Data augmentation for Baseline


    ###---Total Finetune StageII-------
    ##************ HCP 12 WM tracts **************
    ##-----without daug------
    # WEIGHTS_PATH = {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp12/my_custom_experiment_x33/best_weights_ep208.npz',
    # 'cnet':'/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x17/best_weights_ep295.npz'}
    ##-----For CutoutYchange------
    # WEIGHTS_PATH = {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp8/my_custom_experiment_x37/best_weights_ep183.npz',
    # 'cnet':'/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x19/best_weights_ep242.npz'}
    ##-----For CutoutYKeep------
    # WEIGHTS_PATH = {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp8/my_custom_experiment_x38/best_weights_ep63.npz',
    # 'cnet':'/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x21/best_weights_ep268.npz'}
    ##-----For MtractoutYchange------
    # WEIGHTS_PATH = {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x10/best_weights_ep294.npz',
    # 'cnet':'/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x23/best_weights_ep185.npz'}
    ##-----For MtractoutYKeep------
    WEIGHTS_PATH = {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x11/best_weights_ep280.npz',
    'cnet':'/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x24/best_weights_ep259.npz'}



    BATCH_SIZE = 56 # 56/28
    NUM_EPOCHS = 300
    img_size = 144

    EPOCH_MULTIPLIER_train = 1 # 1/6
    EPOCH_MULTIPLIER_val = 1

    WEIGHT_DECAY = 0.0
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True

    lr_s = 'default'
    LR_SCHEDULE_MODE = "min"
    LR_SCHEDULE_PATIENCE = 20

    RESET_LAST_LAYER = False
    DIM = "2D"  # 2D | 3D
    UNET_NR_FILT = 64
    SLICE_DIRECTION = "y"  # x | y | z  ("combined" needs z)
    TRAINING_SLICE_DIRECTION = "xyz"  # y | xyz
    LOSS_FUNCTION = "default"  # default | soft_batch_dice
    OPTIMIZER = "Adamax"
    LOSS_WEIGHT = None  # None = no weighting
    LOSS_WEIGHT_LEN = -1  # -1 = constant over all epochs
    BATCH_NORM = False
    USE_DROPOUT = False
    DROPOUT_SAMPLING = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "My_experiment/best_weights_ep64.npz")
    SAVE_WEIGHTS = True
    TYPE = "single_direction"  # single_direction | combined
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []
    TRAIN = True
    TEST = True
    SEGMENT = True
    GET_PROBS = False
    OUTPUT_MULTIPLE_FILES = False
    UPSAMPLE_TYPE = "bilinear"  # bilinear | nearest
    BEST_EPOCH_SELECTION = "f1"  # f1 | loss
    METRIC_TYPES = ["loss", "f1_macro"]
    FP16 = True
    PEAK_DICE_THR = [0.95]
    PEAK_DICE_LEN_THR = 0.05
    FLIP_OUTPUT_PEAKS = False  # flip peaks along z axis to make them compatible with MITK
    USE_VISLOGGER = False
    SEG_INPUT = "Peaks"  # Gradients | Peaks
    NR_SLICES = 1
    PRINT_FREQ = 10


    # data augmentation
    NORMALIZE_DATA = True
    NORMALIZE_PER_CHANNEL = False
    BEST_EPOCH = 0
    VERBOSE = True
    CALC_F1 = True
    ONLY_VAL = False
    TEST_TIME_DAUG = False
    PAD_TO_SQUARE = True
    INPUT_RESCALING = False  # Resample data to different resolution (instead of doing in preprocessing))

    DATA_AUGMENTATION = True
    DAUG_SCALE = True
    DAUG_NOISE = True
    DAUG_NOISE_VARIANCE = (0, 0.05)
    DAUG_ELASTIC_DEFORM = True
    DAUG_ALPHA = (90., 120.)
    DAUG_SIGMA = (9., 11.)
    DAUG_RESAMPLE = False  # does not improve validation dice (if using Gaussian_blur) -> deactivate
    DAUG_RESAMPLE_LEGACY = False  # does not improve validation dice (at least on AutoPTX) -> deactivate
    DAUG_GAUSSIAN_BLUR = True
    DAUG_BLUR_SIGMA = (0, 1)
    DAUG_ROTATE = False
    DAUG_ROTATE_ANGLE = (-0.2, 0.2)  # rotation: 2*np.pi = 360 degree  (0.4 ~= 22 degree, 0.2 ~= 11 degree))
    DAUG_MIRROR = False
    DAUG_FLIP_PEAKS = False
    SPATIAL_TRANSFORM = "SpatialTransform"  # SpatialTransform|SpatialTransformPeaks
    P_SAMP = 1.0
    DAUG_INFO = "-"
    INFO = "-"

    # for inference
    PREDICT_IMG = False
    PREDICT_IMG_OUTPUT = None
    TRACTSEG_DIR = "tractseg_output"
    KEEP_INTERMEDIATE_FILES = False
    CSD_RESOLUTION = "LOW"  # HIGH | LOW
    NR_CPUS = -1
