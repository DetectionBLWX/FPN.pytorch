'''
Function:
	used to get config file for specified dataset and backbone.
Author:
	Charles
'''
def getCfgByDatasetAndBackbone(datasetname, backbonename):
	if [datasetname, backbonename] == ['coco', 'resnet101']:
		import cfgs.cfg_coco_resnet101 as cfg
		cfg_file_path = 'cfgs/cfg_coco_resnet101'
	elif [datasetname, backbonename] == ['coco', 'resnet50']:
		import cfgs.cfg_coco_resnet50 as cfg
		cfg_file_path = 'cfgs/cfg_coco_resnet50'
	elif [datasetname, backbonename] == ['coco', 'resnet34']:
		import cfgs.cfg_coco_resnet34 as cfg
		cfg_file_path = 'cfgs/cfg_coco_resnet34'
	elif [datasetname, backbonename] == ['coco', 'resnet18']:
		import cfgs.cfg_coco_resnet18 as cfg
		cfg_file_path = 'cfgs/cfg_coco_resnet18'
	else:
		raise ValueError('Can not find cfg file for dataset <%s> and backbone <%s>...' % (datasetname, backbonename))
	return cfg, cfg_file_path