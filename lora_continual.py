import torch
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
import pickle

import compressai

from Iwildcam_Pretrain import Autoencoder, IWildCamDataset, CompressaiWrapper
from lora_modules import LoRAConv2d, LoRALinear, LoRAConvTranspose2d

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

CHECKPOINT_PATH = "./" #"checkpoints/"
HEIGHT = 96 * 2
WIDTH = 160 * 2
PRECISION = 32
QUANTIZATION_PRECISION = None
DO_CACHING = False
DATASET_ROOT = "/data/vision/beery/scratch/data/iwildcam_unzipped"


loraize_encoder = False  # no reason really to do this unless we want to reduce memory footprint
loraize_decoder = True 

lora_config = {
    torch.nn.Conv2d: {
        'cls': LoRAConv2d,
        'config': {
            'alpha': 8,
            'rank': 4,
            'rank_for': 'channels',
            'delta_bias': False ,# True # TODO what does this do
            'precision': 32
        }
    },
    torch.nn.Linear: {
        'cls': LoRALinear,
        'config': {
            'rank': 4,
            'alpha': 2,
            'delta_bias':  False, # True # TODO what does this do
            'precision': 32
        }
    },
    torch.nn.ConvTranspose2d: {
        'cls': LoRAConvTranspose2d,
        'config': {
            'alpha': 8,
            'rank': 4,
            'rank_for': 'channels',
            'delta_bias': False, # True # TODO what does this do
            'precision': 32
        }
    }
}

def get_lora_model(latent_dim=256, lora_precision=32):
    # load two copies so we can lora-ize one
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, f"iwildcam_{latent_dim}.ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print(f"Found pretrained model for latent dim {latent_dim}, loading...")
    #     _model = Autoencoder.load_from_checkpoint(pretrained_filename)
    #     model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # else:
    #     raise Exception

    # model = CompressaiWrapper(compressai.models.ScaleHyperprior(N=192, M=8))
    
    # _model = CompressaiWrapper(compressai.models.ScaleHyperprior(N=192, M=8))
    # model = CompressaiWrapper(compressai.models.ScaleHyperprior(N=192, M=8))
    pretrained_filename = '8-epoch=04-val_loss=0.01.ckpt'
    _model = CompressaiWrapper.load_from_checkpoint(pretrained_filename)
    model = CompressaiWrapper.load_from_checkpoint(pretrained_filename)

    # if loraize_encoder:
    #     for i, module in enumerate(_model.encoder.net.children()): 
    #         if type(module) in lora_config.keys():
    #             lora_cls = lora_config[type(module)]['cls']
    #             lora_params = lora_config[type(module)]['config']
    #             model.encoder.net[i] = lora_cls(module, lora_params) # automatically freezes old parameters
    #             model.encoder.net[i].enable_adapter()                # but we need to turn on the adapter path

    # if loraize_decoder:
    #     for i, module in enumerate(_model.decoder.linear.children()): 
    #         if type(module) in lora_config.keys():
    #             lora_cls = lora_config[type(module)]['cls']
    #             lora_params = lora_config[type(module)]['config']
    #             model.decoder.linear[i] = lora_cls(module, lora_params) # automatically freezes old parameters
    #             model.decoder.linear[i].enable_adapter()                # but we need to turn on the adapter path

    #     for i, module in enumerate(_model.decoder.net.children()): 
    #         if type(module) in lora_config.keys():
    #             lora_cls = lora_config[type(module)]['cls']
    #             lora_params = lora_config[type(module)]['config']
    #             model.decoder.net[i] = lora_cls(module, lora_params) # automatically freezes old parameters
    #             model.decoder.net[i].enable_adapter()                # but we need to turn on the adapter path

    if loraize_decoder:
        # for i, module in enumerate(_model.decoder.linear.children()): 
        #     if type(module) in lora_config.keys():
        #         lora_cls = lora_config[type(module)]['cls']
        #         lora_params = lora_config[type(module)]['config']
        #         model.decoder.linear[i] = lora_cls(module, lora_params) # automatically freezes old parameters
        #         model.decoder.linear[i].enable_adapter()                # but we need to turn on the adapter path

        # for i, module in enumerate(_model.decoder.net.children()): 
        #     if type(module) in lora_config.keys():
        #         lora_cls = lora_config[type(module)]['cls']
        #         lora_params = lora_config[type(module)]['config']
        #         model.decoder.net[i] = lora_cls(module, lora_params) # automatically freezes old parameters
        #         model.decoder.net[i].enable_adapter()  
    
        for name, param in model.model.g_s.named_parameters():
            param.requires_grad = False
        for name, param in model.model.h_s.named_parameters():
            param.requires_grad = False

        print("Fine-tuning:")
        for i, module in enumerate(_model.model.g_s.children()): 
            if type(module) in lora_config.keys():
                print("lora-izing", module)
                lora_cls = lora_config[type(module)]['cls']
                lora_params = lora_config[type(module)]['config']
                model.model.g_s[i] = lora_cls(module, lora_params) # automatically freezes old parameters
                model.model.g_s[i].enable_adapter()                # but we need to turn on the adapter path

        for i, module in enumerate(_model.model.h_s.children()): 
            if type(module) in lora_config.keys():
                print("lora-izing", module)
                lora_cls = lora_config[type(module)]['cls']
                lora_params = lora_config[type(module)]['config']
                model.model.h_s[i] = lora_cls(module, lora_params) # automatically freezes old parameters
                model.model.h_s[i].enable_adapter()                # but we need to turn on the adapter path


    del _model # get rid of the clone
    return model

our_test_set_ids = [
                    # 292, 181, 
                    430, 
                    # 20, 4
                    ]
latent_dim = 256

results = []

# cache the ones we will use
test_set = IWildCamDataset(Path("/data/vision/beery/scratch/data/iwildcam_unzipped"), split="test")
test_set.data['images'] = [i for i in test_set.data['images'] if i['location'] in our_test_set_ids]
test_set.cache_on_device_(device)

for loc_id in our_test_set_ids:
    print("Testing for location", loc_id)
    
    test_set = IWildCamDataset(Path("/data/vision/beery/scratch/data/iwildcam_unzipped"), split="test")
    test_set.data['images'] = [i for i in test_set.data['images'] if i['location'] in our_test_set_ids]
    test_set.cache_on_device_(device)
    test_set._cache = test_set._cache[[idx for idx, i in enumerate(test_set.data['images']) if i['location'] == loc_id]]
    test_set.data['images'] = [i for i in test_set.data['images'] if i['location'] == loc_id]
    seq_ids = list(set([ im['seq_id'] for im in test_set.data['images'] ]))

    # TODO SORT THE SEQUENCES BY DATETIME!

    seq_results = []
    for i in tqdm(range(0, len(seq_ids), 100)): # for last 2 locations, just go every 100 sequences
        seqs_until_now = seq_ids[:i+1]
        test_set = IWildCamDataset(Path("/data/vision/beery/scratch/data/iwildcam_unzipped"), split="test")
        test_set.data['images'] = [i for i in test_set.data['images'] if i['location'] in our_test_set_ids]
        test_set.cache_on_device_(device)
        test_set._cache = test_set._cache[[idx for idx, i in enumerate(test_set.data['images']) if i['location'] == loc_id and i['seq_id'] in seqs_until_now]]
        test_set.data['images'] = [ i for i in test_set.data['images'] if i['location'] == loc_id and i['seq_id'] in seqs_until_now ]
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

        model = get_lora_model(latent_dim)
        print(model)
        
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, f"iwildcam_loc_{loc_id}"),
            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
            precision=PRECISION,
            devices=1,
            max_epochs=100,
            callbacks=[
                ModelCheckpoint(save_weights_only=True),
                LearningRateMonitor("epoch"),
                EarlyStopping(monitor="val_loss", mode="min") # add early stopping
            ]
        )

        # Train the model
        # Overfit to the test set
        trainer.fit(model, test_loader, test_loader)
        save_path = os.path.join(CHECKPOINT_PATH, f"iwildcam_latent_dim={latent_dim}_lora_loc={loc_id}.ckpt")
        trainer.save_checkpoint(save_path)

        # test_reconstruction_error = trainer.test(model=model, dataloaders=test_loader)[0]['test_loss']
        # test_reconstruction_error /= len(test_loader) # TODO: I think what we get from test() is a sum over batches?

        test_metrics = trainer.test(model=model, dataloaders=test_loader)[0]

        # Store results
        seq_results.append((loc_id, seq_ids[i], len(test_set.data['images']), test_metrics))

        # release memory
        del test_set
        del model
        del trainer
        torch.cuda.empty_cache()

        # save intermediate results in case everything dies
        pickle.dump(seq_results, open(f'{loc_id}_results_hyperprior.pkl', 'wb'))

    results.append(seq_results)
    
pickle.dump(results, open('all_results_hyperprior.pkl', 'wb'))