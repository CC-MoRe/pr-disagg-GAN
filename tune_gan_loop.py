

import os

import json
import numpy as np



clip_value = 0.01
batch_size = 128
n_epochs = 500
latent_dim=100
clip_value=0.01
i_config = 0
for n_hidden in [4,10,50,100]:
    for hidden_size in [512,1024]:
        for drop_prob in [0,0.5,0.9]:
            for n_disc in [5,]:
                    config_all = {
                        'config':{'disc_config':{'n_hidden':n_hidden, 'hidden_size':hidden_size, 'leakyrelu_alpha':0.2, 'drop_prob':drop_prob},
                               'gen_config':{'n_hidden':n_hidden, 'hidden_size':hidden_size,'latent_dim':latent_dim, 'leakyrelu_alpha':0.2,}
                        },
                        'n_disc':n_disc,
                        'i_config':i_config,
                        'n_epochs':n_epochs,
                        'batch_size':batch_size,
                        'clip_value':clip_value,
                    }
                    os.system(f'''sbatch -A snic2019-1-2 --time=03-00:00:00 -N 1 -o slurm_training{i_config:04d}.out <<EOF
#! /bin/bash
/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-svd-env/bin/python pr_disagg_batch.py '{json.dumps(config_all)}'
EOF
''')
                    i_config = i_config+1

print(i_config)