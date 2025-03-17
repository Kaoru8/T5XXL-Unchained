
import os, copy

import torch, safetensors

from safetensors.torch import save_file

def convertModel(oldPath, quantize=False):
	oldEmbedSize, newEmbedSize = 32128, 69328
	addName = "-unchained"
	if quantize == True or "fp8" in os.path.basename(oldPath): addName+="-f8"
	else: addName+="-f16"
	saveName = os.path.basename(oldPath).replace("-fp8","").replace("_fp16","").replace(".safetensors",addName+".safetensors")
	if os.path.exists(saveName):
		print("Model already exists at "+saveName+", aborting")
	else:
		print("Converting",os.path.basename(oldPath),"and saving as",saveName)
		tensorDict = {}
		with safetensors.safe_open(oldPath, framework="pt", device=0) as f:
			for k in f.keys():
				tensor_slice = f.get_slice(k)
				tSize = [x for x in tensor_slice.get_shape()]
				if tSize == [oldEmbedSize,4096]:
					print("\tExpanding layer",k)
					tensor = f.get_tensor(k).to(0)
					initDtype = copy.deepcopy(tensor.dtype)
					d = torch.randn([newEmbedSize-oldEmbedSize,4096],dtype=tensor.dtype).to(0)
					newTensor = torch.cat((tensor.to(0),d.to(0)))
					if initDtype != newTensor.dtype: newTensor = newTensor.to(dtype=initDtype)
					tensorDict[k] = newTensor
				else: tensorDict[k] = f.get_tensor(k)
				if quantize == True and tensorDict[k].dtype != torch.float8_e4m3fn: tensorDict[k] = tensorDict[k].to(dtype=torch.float8_e4m3fn)
		save_file(tensorDict, saveName, metadata={"format":"pt"})






convertModel("E:/ComfyUI/ComfyUI/models/clip/t5xxl_fp16.safetensors", quantize=False)