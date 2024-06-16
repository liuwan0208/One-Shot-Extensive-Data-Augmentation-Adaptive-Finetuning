import os

# ####**********************second Stage***************************
# # #-----------proposed-------------
# # #*****finetune**************
os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange'    --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x189/best_weights_ep172.npz'} ")
os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange'    --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x189/best_weights_ep172.npz'} ")



####**********************First Stage***************************
#-----------proposed-------------
#*****warmup**************
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/FourCombined_400'    --train_sub '845458' --fix_other_layer True --use_daug True")

















##########################--------------------------------------------------------------HCP_2.5_34-----------------------------------------------------------------------


# ####**********************second Stage***************************
# #------------IFT-----------------
# #*******warmup*******
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange' --train_sub '845458' --weight_path {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp12/my_custom_experiment_x33/best_weights_ep208.npz'} ")

# # #-----------proposed-------------
# # #*****warmup**************
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange'    --train_sub '845458' --weight_path {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp8/my_custom_experiment_x37/best_weights_ep183.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/CutoutYKeep'      --train_sub '845458' --weight_path {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp8/my_custom_experiment_x38/best_weights_ep63.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYChange' --train_sub '845458' --weight_path {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x10/best_weights_ep294.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYKeep'   --train_sub '845458' --weight_path {'seg':'/data6/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x11/best_weights_ep280.npz'} ")












##########################--------------------------------------------------------------HCP_1.25_34-----------------------------------------------------------------------
# # ####**********************second Stage***************************
# # #------------IFT-----------------
# # #*******warmup*******
# os.system("python ExpRunner3 --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange' --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x51/best_weights_ep152.npz'} ")

# # #-----------proposed-------------
# # #*****warmup**************
# os.system("python ExpRunner3 --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange'    --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x52/best_weights_ep232.npz'} ")
# os.system("python ExpRunner3 --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYKeep'      --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x55/best_weights_ep110.npz'} ")
# os.system("python ExpRunner3 --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYChange' --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x56/best_weights_ep97.npz'} ")
# os.system("python ExpRunner3 --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYKeep'   --train_sub '845458' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x57/best_weights_ep218.npz'} ")


####**********************First Stage***************************
#-------------CFT----------------
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange' --train_sub '845458' ")

#------------IFT-----------------
#*******warmup*******
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange' --train_sub '845458' --fix_other_layer True")

#-----------proposed-------------
#*****warmup**************
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYChange'    --train_sub '845458' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/CutoutYKeep'      --train_sub '845458' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYChange' --train_sub '845458' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_1.25mm_341k/HCP_daug/Oneshot_845458_12/MtractoutYKeep'   --train_sub '845458' --fix_other_layer True --use_daug True")














##########################--------------------------------------------------------------HCP_1.25_270 multiple one shot -----------------------------------------------------------------------

#####**********************second Stage***************************
##------------IFT-----------------
#*******warmup*******
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYChange' --train_sub '837964' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x7/best_weights_ep201.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYChange' --train_sub '833249' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x8/best_weights_ep116.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYChange' --train_sub '833148' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x9/best_weights_ep137.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYChange' --train_sub '826454' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x10/best_weights_ep263.npz'} ")


# # # ##-----------proposed------------
# # # ##*****warmup**************
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYChange'    --train_sub '837964' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x11/best_weights_ep173.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYKeep'      --train_sub '837964' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x14/best_weights_ep59.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/MtractoutYChange' --train_sub '837964' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x17/best_weights_ep85.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/MtractoutYKeep'   --train_sub '837964' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x21/best_weights_ep69.npz'} ")

# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYChange'    --train_sub '833249' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x22/best_weights_ep251.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYKeep'      --train_sub '833249' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x23/best_weights_ep59.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/MtractoutYChange' --train_sub '833249' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x24/best_weights_ep175.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/MtractoutYKeep'   --train_sub '833249' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x25/best_weights_ep230.npz'} ")

# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYChange'    --train_sub '833148' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x13/best_weights_ep139.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYKeep'      --train_sub '833148' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x16/best_weights_ep72.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/MtractoutYChange' --train_sub '833148' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x20/best_weights_ep98.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/MtractoutYKeep'   --train_sub '833148' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x26/best_weights_ep138.npz'} ")

# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYChange'    --train_sub '826454' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x12/best_weights_ep245.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYKeep'      --train_sub '826454' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x15/best_weights_ep261.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/MtractoutYChange' --train_sub '826454' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x18/best_weights_ep128.npz'} ")
# os.system("python ExpRunner_step2 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/MtractoutYKeep'   --train_sub '826454' --weight_path {'seg':'/data7/wanliu/TractSeg_Run/hcp_exp9/my_custom_experiment_x19/best_weights_ep147.npz'} ")




#####**********************First Stage***************************
##-------------CFT---------------
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYChange' --train_sub '837964' ")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYChange' --train_sub '833249' ")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYChange' --train_sub '833148' ")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYChange' --train_sub '826454' ")


##------------IFT-----------------
##*******warmup*******
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYChange' --train_sub '837964' --fix_other_layer True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYChange' --train_sub '833249' --fix_other_layer True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYChange' --train_sub '833148' --fix_other_layer True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYChange' --train_sub '826454' --fix_other_layer True")


##-----------proposed------------
##*****warmup**************
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYChange'    --train_sub '837964' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/CutoutYKeep'      --train_sub '837964' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/MtractoutYChange' --train_sub '837964' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_837964_12/MtractoutYKeep'   --train_sub '837964' --fix_other_layer True --use_daug True")

# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYChange'    --train_sub '833249' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/CutoutYKeep'      --train_sub '833249' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/MtractoutYChange' --train_sub '833249' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833249_12/MtractoutYKeep'   --train_sub '833249' --fix_other_layer True --use_daug True")

# os.system("python ExpRunner1 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYChange'    --train_sub '833148' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner1 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/CutoutYKeep'      --train_sub '833148' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner1 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/MtractoutYChange' --train_sub '833148' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner1 --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_833148_12/MtractoutYKeep'   --train_sub '833148' --fix_other_layer True --use_daug True")

# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYChange'    --train_sub '826454' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/CutoutYKeep'      --train_sub '826454' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/MtractoutYChange' --train_sub '826454' --fix_other_layer True --use_daug True")
# os.system("python ExpRunner --datapath '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_826454_12/MtractoutYKeep'   --train_sub '826454' --fix_other_layer True --use_daug True")

