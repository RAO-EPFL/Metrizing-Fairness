import bias_eval
from pickle import dump
from tqdm import tqdm

for seed in tqdm(range(20)):
    for target_batchsize in [4, 8, 16, 32, 64, 128, 256, 512]:
        results, path,_ = bias_eval.train_metrics_debiased(target_batchsize, seed=seed)
        with open(f'results\\results_debiased_seed{seed}_bs{target_batchsize}.pickle', 'wb') as handle:
            dump(results, handle)
        with open(f'results\\path_debiased_seed{seed}_bs{target_batchsize}.pickle', 'wb') as handle:
            dump(path, handle)
            
        results, path,_ = bias_eval.train_metrics_fullbias(target_batchsize, seed=seed)
        with open(f'results\\results_fullbias_seed{seed}_bs{target_batchsize}.pickle', 'wb') as handle:
            dump(results, handle)
        with open(f'results\\path_fullbias_seed{seed}_bs{target_batchsize}.pickle', 'wb') as handle:
            dump(path, handle)
            
        results, path,_ = bias_eval.train_metrics_noreg(target_batchsize, seed=seed)
        with open(f'results\\results_seed{seed}_bs{target_batchsize}_noreg.pickle', 'wb') as handle:
            dump(results, handle)
        with open(f'results\\path_seed{seed}_bs{target_batchsize}_noreg.pickle', 'wb') as handle:
            dump(path, handle)