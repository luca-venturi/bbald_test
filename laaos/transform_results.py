import pickle
import argparse
import os.path

def save_results_our_format(exp_name, run):
    exec('from {}{} import store as data'.format(exp_name, run), None, globals())
    acc = []
    for x in data['iterations']:
        acc.append(x['test_metrics']['accuracy'])
    return acc

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='mnfull', type=str)
    parser.add_argument('--runs', default=[0], type=int, nargs='+')
    parser.add_argument('--name_to_save', default='mnfull1_batchbald', type=str)

    args = parser.parse_args()

    filepath = 'results/{}'.format(args.name_to_save)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as rfile:
            results = pickle.load(rfile)
    else:
        results = {'runs':[]}

    for run in args.runs:
        if run not in results['runs']:
            results['runs'].append(run)
        to_add = save_results_our_format(args.experiment, run)
        results[run, 'accuracies_queries'] = to_add

    with open(filepath, 'wb') as wfile:
        pickle.dump(results, wfile)
