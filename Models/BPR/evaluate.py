import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item_i, item_j in test_loader:
		user = user.cuda()
		item_i = item_i.cuda() # 무작위 100개 
		item_j = item_j.cuda() # not useful when testing

		# print ("--- item_i 1")
		# print (item_i)
  
		prediction_i, prediction_j = model(user, item_i, item_j)
  
		# print ("Prediction_i.shape", prediction_i.shape)
		_, indices = torch.topk(prediction_i, top_k)
		# print ("TOP k finished")
    
		recommends = torch.take(
				item_i, indices).cpu().numpy().tolist()
  
		gt_item = item_i[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)
