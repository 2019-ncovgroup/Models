# Tensorflow Estimator implementation

## Data prep and Training
1. convert data to Tensorflow Records format
```
python to_tfrec.py --data_file ../data/pl.ena.csv --out_dir ../data/plpro.reg/
```

2. Training
```
# GPU mode
python train_gpu.py --mode train --data_dir ../data/plpro.reg --model_dir model_plpro.reg

# CS1
srun_train python train_cs1.py --mode train --cs_ip 10.80.0.100 --data_dir /data/shared/hsyoo/data/covid/plpro.reg/ --model_dir model_plpro.reg
```

3. Evaluation
```
# GPU mode
python train_gpu.py --mode eval --data_dir ../data/plpro.reg --model_dir model_plpro.reg

# CS1
python train_cs1.py --mode eval --data_dir /data/shared/hsyoo/data/covid/plpro.reg/ --model_dir model_plpro.reg
```


## Model conversion
Estimator model will produce checkpoints and we can convert the weights to keras format.

```
python cs1_to_keras_model.py --model_dir model_plpro.reg/ --checkpoint model.ckpt-137400 --out model_plpro.reg.h5
```
