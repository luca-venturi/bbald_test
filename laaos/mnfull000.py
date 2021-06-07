store = {}
store['args']={'experiment_task_id': 'mnfull000', 'experiments_laaos': None, 'experiment_description': 'Trying stuff..', 'batch_size': 8, 'scoring_batch_size': 256, 'test_batch_size': 256, 'validation_set_size': 128, 'early_stopping_patience': 100, 'epochs': 100, 'epoch_samples': 5056, 'num_inference_samples': 20, 'available_sample_k': 20, 'target_num_acquired_samples': 220, 'target_accuracy': 0.98, 'no_cuda': False, 'quickquick': False, 'seed': 1, 'log_interval': 10, 'initial_samples_per_class': 2, 'initial_samples': None, 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'dataset': 'DatasetEnum.mnist10kconv', 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 20, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False}
store['cmdline']=['run_experiment.py', '--batch_size', '8', '--epochs', '100', '--early_stopping_patience', '100', '--num_inference_samples', '20', '--available_sample_k', '20', '--target_num_acquired_samples', '220', '--initial_samples_per_class', '2', '--acquisition_method', 'multibald', '--dataset', 'mnist10kconv', '--experiment_task_id', 'mnfull000']
store['iterations']=[]
store['initial_samples']=[5845, 5321, 4470, 4246, 9002, 9151, 1463, 1007, 8065, 8238, 2043, 2656, 6415, 6947, 7851, 7136, 3326, 3711, 880, 948]