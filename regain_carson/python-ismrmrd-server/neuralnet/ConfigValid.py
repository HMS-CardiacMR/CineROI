import os
import datetime

import torch
#import torch.backends.cudnn as cudnn
from Models import GeneratorRRDB
import sys

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(123)
cudnn.benchmark = True
cudnn.deterministic = False
#device = torch.device("cuda:0")
device = torch.device("cuda:0")
# Runing mode.
mode = "validate"
scale_factor = 4


# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":

    generator = GeneratorRRDB(1, filters=64, num_res_blocks=23).to(device)
#    net.load_state_dict(torch.load("./results/G_epoch71.pth", map_location=device))

    generator.load_state_dict(torch.load("./samples/G_epoch200.pth", map_location=device))
    sys.path.append("/mnt/alp/users/siyeopyoon/")
    print(os.getcwd())

    #    sr_dir = "./data/test3/Ph_0915_PE_ExtreamTest/Pairs"
    #    root_dir = "/mnt/alp/users/siyeopyoon/F02Data/1_SR/Pair0930_AfterIQA_SameSize"
    #    sr_dir = "./data/Pair0930_AfterIQA_SameSize_epoch75"

    root_dir = "/mnt/alp/users/siyeopyoon/F02Data/1_SR/Data-20211018"
    sr_dir = "./data/Data-20211018_200Epoch"

    #    lr_dir = "P:/ALP/Users/SiyeopYoon/F02Data/1_SR/V_0916_ExampleTest_0929_databeforelocalfilter_SAXGRAPPA/TEST_LR_mag"
    lr_dir = root_dir+"/Test_LR_mag"
    hr_dir = root_dir+"/Test_HR_mag"
#    hr_dir = "P:/ALP/Users/SiyeopYoon/F02Data/1_SR/V_0916_ExampleTest_0929_databeforelocalfilter_SAXGRAPPA/TEST_HR_mag"
    pad_dir = "P:/ALP/Users/SiyeopYoon/F02Data/1_SR/Ph_0915_PE1/ZeroPad_mag"



#    lr_dir = "P:/ALP/Users/SiyeopYoon/F02Data/1_PairForSuperResolution/FunTest/LR_mag"
#    hr_dir = "P:/ALP/Users/SiyeopYoon/F02Data/1_PairForSuperResolution/FunTest/HR_mag"