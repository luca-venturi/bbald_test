from aux import *

def display_curves(file_, curves, criteria, string):
	colors = []
	for c in criteria:
		label = c
		line = '-'
		if 'diffusion' in c:
			label = 'diffusion'
			line = '--'
		pq = 20
		queries = pq + pq * np.arange(len(curves[c, 'mean']))
		start_at = 0
		end_at = 40
		ebar = plt.errorbar(queries[start_at:end_at+1], curves[c, 'mean'][start_at:end_at+1], yerr=curves[c, 'std'][start_at:end_at+1], label=label, linestyle=line)
		colors.append(mpl.colors.to_rgba(ebar[0].get_color()))
	print(colors)
	#plt.xlabel('# labeled data points queried', fontsize=11)
	#plt.ylabel('accuracy on test set', fontsize=11)

	tickssize = 17
	plt.xticks(queries[start_at:end_at+1], fontsize=tickssize)
	plt.yticks(fontsize=tickssize)
	#plt.legend(prop={'size': 13})
	fig = plt.gcf()
	ax = plt.gca()
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	fig.set_size_inches(5,5)
	#plt.show()
	plt.savefig('{}_{}.pdf'.format(file_, string))

if __name__=='__main__':
	runs = {'random':[0, 1, 2, 3, 4], 'uncertainty':[0, 1, 2, 3, 4], 'coreset':[0, 1, 2, 3, 4], 'bayes-entropy':[0, 1, 2, 3, 4], 'entropy':[0, 1, 2, 3, 4], 'margin':[0, 1, 2, 3, 4], 'bayes-uncertainty':[0, 1, 2, 3, 4], 'diffusion-one-vs-all-min-k10-t5-sa1-mode10':[0, 1, 2, 3, 4], 'badge':[0,1,2,3,4], 'batchbald':[0]}
	# svhntpre1 diffusion-one-vs-all-min-k20-t5-sa1-mode10-softlabel1-atperc0.1 explore(0, 10) pq 200
	# mnconv1 diffusion-one-vs-all-min-k10-t5-sa1-mode1 explore(0,10) pq 20 - refine(10,49)
	# mnfull1 diffusion-one-vs-all-min-k10-t5-sa1-mode10 explore(0,40) pq 20
	# cifar10pre1 diffusion-one-vs-all-min-k20-t4-sa1-mode10-atperc0.1 explore(0,10) pq 200
	files = ['mnfull1']
	curves = {}
	string = input('Additional filename string: ')
	for i,c in enumerate(runs):
		with open('data_results/results/'+files[0]+'_'+c, 'rb') as rfile:
			results = pickle.load(rfile)
			curves[c, 'mean'] = np.mean(np.array([results[r,'accuracies_queries'][:] for r in runs[c]]), axis=0)
			curves[c, 'std'] = np.std(np.array([results[r,'accuracies_queries'][:] for r in runs[c]]), axis=0)
	display_curves(files[0], curves, runs.keys(), string)
