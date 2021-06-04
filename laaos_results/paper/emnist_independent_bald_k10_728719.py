store = {}
store['args']={'name': 'emnist_independent_bald_k10_728719', 'available_sample_k': 5, 'num_inference_samples': 10, 'seed': 728719, 'acquisition_method': 'AcquisitionMethod.independent', 'experiment_description': 'EMNIST with b5 and k10, k100 with both BALD and BatchBALD', 'type': 'AcquisitionFunction.bald', 'batch_size': 64, 'scoring_batch_size': 512, 'test_batch_size': 512, 'validation_set_size': 16384, 'early_stopping_patience': 3, 'epochs': 40, 'epoch_samples': 20224, 'target_accuracy': 0.85, 'target_num_acquired_samples': 300, 'log_interval': 20, 'dataset': 'DatasetEnum.emnist', 'initial_samples': [], 'experiment_task_id': 16, 'experiments_laaos': './experiment_configs/emnist_bbb/configs.py', 'no_cuda': False, 'quickquick': False, 'initial_samples_per_class': 2}
store['cmdline']=['./src/ignite_mnist.py', '--experiment_task_id=16', '--experiments_laaos=./experiment_configs/emnist_bbb/configs.py']
store['iterations']=[]
store['initial_samples']=[]
store['iterations'].append({'num_epochs': 0, 'test_metrics': {'accuracy': 0.020904255319148937, 'nll': 3.864540234829517}, 'chosen_samples': [73715, 7843, 105213, 107454, 50309], 'chosen_samples_score': [0.012549799308180454, 0.012559230625629247, 0.012843641266226502, 0.01257765591144544, 0.01279014758765662], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.02297872340425532, 'nll': 63.88668759285135}, 'chosen_samples': [23066, 14356, 85841, 91637, 33087], 'chosen_samples_score': [0.4633615893944171, 0.46659336612908353, 0.492948108147504, 0.5113997982267358, 0.5010688015825853], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.045159574468085106, 'nll': 36.34060002103765}, 'chosen_samples': [40730, 71611, 3935, 28995, 78570], 'chosen_samples_score': [1.2353099106001713, 1.2365620877085464, 1.289190802175931, 1.3045343010972257, 1.2377572854177807], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.06659574468085107, 'nll': 44.05347753971181}, 'chosen_samples': [510, 112178, 92790, 41261, 50270], 'chosen_samples_score': [1.278024800942919, 1.3020895832663866, 1.3064513270561346, 1.2997288980489876, 1.2782682069041003], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.0750531914893617, 'nll': 39.34572762671937}, 'chosen_samples': [112007, 82974, 94049, 6197, 43192], 'chosen_samples_score': [1.4215651205003907, 1.4410385272071462, 1.44508513186844, 1.4604667675113778, 1.4536805923803895], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.0950531914893617, 'nll': 32.639400008181305}, 'chosen_samples': [34472, 23065, 39026, 16740, 63264], 'chosen_samples_score': [1.5051333190715226, 1.5103096964915892, 1.51942415837067, 1.561008238326571, 1.565485070593508], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.1025531914893617, 'nll': 30.47116108671148}, 'chosen_samples': [81350, 57314, 9673, 30326, 35100], 'chosen_samples_score': [1.5541757678884938, 1.5542568076892804, 1.5733367612529268, 1.5955625230512067, 1.6449160388980943], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.11696808510638299, 'nll': 30.886893985829456}, 'chosen_samples': [112499, 19397, 104197, 33862, 86199], 'chosen_samples_score': [1.6385510425056977, 1.6454812750286039, 1.6466975035414158, 1.6476338893590834, 1.740343946269845], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.14622340425531916, 'nll': 24.21684246631379}, 'chosen_samples': [6276, 10456, 57526, 78829, 70417], 'chosen_samples_score': [1.5924863632032256, 1.6095180204669328, 1.616524575517059, 1.6259400441002188, 1.6483755381129044], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.1529787234042553, 'nll': 22.07460325200507}, 'chosen_samples': [21288, 65682, 91380, 13919, 4192], 'chosen_samples_score': [1.6621753931361356, 1.6676888309601594, 1.7318538207910266, 1.7168386492797063, 1.7569490655457352], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.1595744680851064, 'nll': 20.69380258438435}, 'chosen_samples': [24528, 83042, 64082, 29175, 4243], 'chosen_samples_score': [1.462557294178111, 1.4671965042462427, 1.4752355587572938, 1.4997547648516125, 1.5274228930461446], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.15867021276595744, 'nll': 19.401814068733376}, 'chosen_samples': [33828, 92096, 91675, 8387, 83212], 'chosen_samples_score': [1.554882562775434, 1.5658644628366813, 1.574648424366738, 1.5765302123566527, 1.585638646765794], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.19877659574468085, 'nll': 21.066711511814848}, 'chosen_samples': [110305, 84327, 32417, 50922, 76657], 'chosen_samples_score': [1.6629437454533211, 1.6647643217730606, 1.6757315168744586, 1.7052959830969923, 1.7211570746151053], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.20042553191489362, 'nll': 20.687437635381173}, 'chosen_samples': [93714, 7836, 14384, 76593, 42506], 'chosen_samples_score': [1.8262138847654064, 1.8529309202490214, 1.8536631982556078, 1.8577875352843671, 1.8955995507767698], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.20957446808510638, 'nll': 19.008824035969187}, 'chosen_samples': [93750, 83838, 14931, 85088, 85522], 'chosen_samples_score': [1.5907997849358764, 1.5985096061513375, 1.6380260373920044, 1.6126194364467263, 1.7768420338530462], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.20707446808510638, 'nll': 17.489202583799973}, 'chosen_samples': [88455, 27985, 6064, 61909, 53714], 'chosen_samples_score': [1.598190769263783, 1.598466348471525, 1.6118549805760356, 1.6588306220833946, 1.6245262768671995], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.22117021276595744, 'nll': 15.985795011317476}, 'chosen_samples': [35747, 67714, 76763, 13067, 101750], 'chosen_samples_score': [1.6088709987785679, 1.6256080130689798, 1.6320786464674963, 1.66971474835504, 1.739218891624765], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.23962765957446808, 'nll': 16.00663659521874}, 'chosen_samples': [17329, 34881, 41281, 372, 28267], 'chosen_samples_score': [1.674132777769474, 1.6880852343053636, 1.7765342898877872, 1.7293109159437137, 1.7418413380821347], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.23361702127659575, 'nll': 16.761993002384266}, 'chosen_samples': [41849, 51563, 8782, 23181, 81098], 'chosen_samples_score': [1.6348127960539791, 1.6521473453098197, 1.7189822638578207, 1.6771604636804924, 1.7507908575463442], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.22122340425531914, 'nll': 17.18456147376527}, 'chosen_samples': [11935, 18020, 68428, 76868, 98164], 'chosen_samples_score': [1.5913703116538083, 1.6150187131559117, 1.632968755733558, 1.6205718158009503, 1.6036608744534084], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.25632978723404254, 'nll': 16.603189463513964}, 'chosen_samples': [81111, 89170, 54554, 35644, 58699], 'chosen_samples_score': [1.7850620279379634, 1.798069777665036, 1.8194783101612118, 1.92516255594316, 1.8508538964783279], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.25638297872340426, 'nll': 15.619791682628875}, 'chosen_samples': [39274, 67176, 27355, 11545, 32930], 'chosen_samples_score': [1.6576241787448962, 1.6722811423696822, 1.7091211953875667, 1.812951353189686, 1.7207239487885846], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.26303191489361705, 'nll': 15.206106674518992}, 'chosen_samples': [66266, 109083, 97751, 99233, 26493], 'chosen_samples_score': [1.5546327421995845, 1.5754710340560432, 1.57831757439359, 1.8134241250907652, 1.5827949007000368], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.27159574468085107, 'nll': 14.031446227215707}, 'chosen_samples': [74387, 63392, 4044, 49939, 72269], 'chosen_samples_score': [1.7048879586345644, 1.71770224258679, 1.7190075600148291, 1.7406623436073878, 1.7269371760150996], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.28569148936170213, 'nll': 12.992672474637944}, 'chosen_samples': [36818, 21336, 22463, 33444, 9943], 'chosen_samples_score': [1.6314495682967922, 1.6616056535663477, 1.6622252874643713, 1.6798002776048566, 1.682088245433024], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.29659574468085104, 'nll': 12.588668346810849}, 'chosen_samples': [59802, 8507, 25130, 71757, 84010], 'chosen_samples_score': [1.5542360827765815, 1.5577291546607017, 1.6072453214846776, 1.571106708303386, 1.5624386594507151], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3091489361702128, 'nll': 11.731236959416815}, 'chosen_samples': [47904, 65719, 109315, 38036, 30476], 'chosen_samples_score': [1.5155610998948275, 1.5288211582288351, 1.5493869926682455, 1.5599034530884168, 1.5544184983674465], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3103723404255319, 'nll': 11.449541117891352}, 'chosen_samples': [65251, 36278, 16191, 74678, 42645], 'chosen_samples_score': [1.6475154883764982, 1.647639042510312, 1.6562677959017393, 1.7127112419652792, 1.6821044449245397], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3125531914893617, 'nll': 11.246867906286361}, 'chosen_samples': [20509, 77010, 110284, 27644, 48480], 'chosen_samples_score': [1.6245088963472334, 1.6272649541375257, 1.6802081166133096, 1.7579297108607863, 1.6336119962704982], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.33111702127659576, 'nll': 10.102602600746966}, 'chosen_samples': [27825, 110032, 13542, 292, 31013], 'chosen_samples_score': [1.6362210863299136, 1.6415335999776763, 1.6605517475239884, 1.6955757462736754, 1.7746357219429905], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3402127659574468, 'nll': 9.553519671014016}, 'chosen_samples': [24558, 18727, 19068, 7853, 100848], 'chosen_samples_score': [1.5361126142976391, 1.5500634753592362, 1.562947631094493, 1.5656517692906307, 1.574490083398162], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3573936170212766, 'nll': 8.440931064524548}, 'chosen_samples': [45475, 19109, 34743, 92097, 74908], 'chosen_samples_score': [1.604705212314146, 1.624731174271656, 1.6271909541008904, 1.6860395704083055, 1.7248616919835118], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.34952127659574467, 'nll': 8.648938877024548}, 'chosen_samples': [69843, 20389, 98223, 72029, 78108], 'chosen_samples_score': [1.5868007725068485, 1.598623407019228, 1.641767883292278, 1.6743149096580172, 1.7045625743610833], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.33787234042553194, 'nll': 10.353457331555955}, 'chosen_samples': [63863, 104456, 41886, 71476, 7641], 'chosen_samples_score': [1.5446961002975832, 1.5463044213337112, 1.55012346137448, 1.6083969639574107, 1.687153537186326], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3601063829787234, 'nll': 8.923032134847437}, 'chosen_samples': [94130, 81290, 29723, 49124, 30112], 'chosen_samples_score': [1.530121846813862, 1.5568660205810483, 1.586469549687553, 1.5676688810869377, 1.6500119209073363], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.36856382978723407, 'nll': 8.140440549647554}, 'chosen_samples': [90943, 55206, 79487, 20132, 8084], 'chosen_samples_score': [1.4528666548635871, 1.4542494520301448, 1.4599561329661763, 1.4960291400364287, 1.5496018715045994], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3720212765957447, 'nll': 8.26905019394895}, 'chosen_samples': [40306, 65189, 7699, 83592, 63657], 'chosen_samples_score': [1.5447231601055602, 1.5587410384616396, 1.6156034782608177, 1.5803412011178957, 1.6212583239884508], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3657446808510638, 'nll': 8.687114001334981}, 'chosen_samples': [67050, 24814, 33612, 46521, 73389], 'chosen_samples_score': [1.524752831905949, 1.5253181004017542, 1.5654981832752712, 1.532143775533541, 1.606552378226389], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.36781914893617024, 'nll': 7.9149345126050585}, 'chosen_samples': [52692, 108074, 37605, 99277, 103154], 'chosen_samples_score': [1.51543581727426, 1.5309290864373146, 1.5355971473982315, 1.541759214576501, 1.5651878723517725], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3803723404255319, 'nll': 8.04324480665491}, 'chosen_samples': [110831, 41486, 46598, 83415, 97195], 'chosen_samples_score': [1.4493227343474009, 1.4502019570289293, 1.4528424331182332, 1.4522753204863, 1.4741532153675494], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.38101063829787235, 'nll': 8.4623864514777}, 'chosen_samples': [108737, 83655, 5498, 111557, 87673], 'chosen_samples_score': [1.6092486861253765, 1.650570760162171, 1.667558400792163, 1.68108283958615, 1.7715812303001037], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.40606382978723404, 'nll': 7.728865886038922}, 'chosen_samples': [76394, 15052, 26085, 51186, 68643], 'chosen_samples_score': [1.60131682103178, 1.6185072831321523, 1.6295119970720173, 1.688151485849008, 1.7613250414420771], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4068617021276596, 'nll': 7.135081371145046}, 'chosen_samples': [44961, 111994, 79195, 35856, 43368], 'chosen_samples_score': [1.4543832386302533, 1.5506238660001697, 1.4825971108864546, 1.459038732853777, 1.5201800523807827], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3891489361702128, 'nll': 7.560692451558214}, 'chosen_samples': [25343, 39261, 78735, 51150, 84119], 'chosen_samples_score': [1.5907194301309793, 1.5950292766176397, 1.599685886081999, 1.6431282040435362, 1.6360009626416185], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.3934042553191489, 'nll': 8.24014813727521}, 'chosen_samples': [2098, 21050, 56962, 77719, 106158], 'chosen_samples_score': [1.6597417119643074, 1.7271903507269528, 1.679510170740599, 1.7327262680849713, 1.75237270270229], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4081382978723404, 'nll': 7.2812851893648185}, 'chosen_samples': [88077, 64101, 56275, 20285, 23698], 'chosen_samples_score': [1.5292224973173312, 1.531699756381268, 1.5373388120868174, 1.5457340078233952, 1.5628804702105596], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4060106382978723, 'nll': 7.229754140732136}, 'chosen_samples': [19971, 22045, 3303, 413, 66841], 'chosen_samples_score': [1.4980587268349603, 1.5089908898787883, 1.502587228511786, 1.5061988522006162, 1.5509855401793873], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4193085106382979, 'nll': 7.613720470996613}, 'chosen_samples': [8247, 93739, 87560, 98248, 87061], 'chosen_samples_score': [1.516840323115312, 1.5297864056029047, 1.523647988781847, 1.5246865847539708, 1.5613862273299701], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.41319148936170214, 'nll': 7.333194373516326}, 'chosen_samples': [49585, 90521, 62201, 104818, 51100], 'chosen_samples_score': [1.5741437415286514, 1.5747410165877844, 1.5754602651826348, 1.59066510787533, 1.5813026447759997], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.41654255319148936, 'nll': 7.05826509840945}, 'chosen_samples': [76291, 58818, 75928, 108540, 101777], 'chosen_samples_score': [1.5614608234253207, 1.5809861840055266, 1.5855104490102843, 1.5880210508843307, 1.6278612925549294], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.42606382978723406, 'nll': 6.6748619282499275}, 'chosen_samples': [10868, 9461, 16710, 65237, 58524], 'chosen_samples_score': [1.5263335074033748, 1.5310668819005062, 1.551841306548289, 1.5653997964429567, 1.5740100355324762], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.41648936170212764, 'nll': 7.191716517184643}, 'chosen_samples': [10599, 34227, 12137, 32595, 38267], 'chosen_samples_score': [1.6002187188332493, 1.603325012558762, 1.6121794600900443, 1.6492533159293963, 1.6285467816064614], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4176595744680851, 'nll': 6.383121298932014}, 'chosen_samples': [18915, 6904, 63756, 86426, 49142], 'chosen_samples_score': [1.4228102956731372, 1.4355826259718616, 1.4425845934224832, 1.4543318256089126, 1.5285224017988908], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4301595744680851, 'nll': 7.027159101202133}, 'chosen_samples': [28667, 82615, 49228, 56051, 110419], 'chosen_samples_score': [1.5694899544232468, 1.6039314911474374, 1.6322207381897884, 1.6415260055259555, 1.584802838369316], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4453191489361702, 'nll': 6.4224159374642875}, 'chosen_samples': [34623, 32082, 45865, 17058, 57623], 'chosen_samples_score': [1.5679865673209026, 1.5979511994651716, 1.5989169733037567, 1.6160246545590893, 1.640290604586645], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4477659574468085, 'nll': 7.0156526691355605}, 'chosen_samples': [294, 9453, 98849, 37737, 85556], 'chosen_samples_score': [1.5573558145711786, 1.5655218764984342, 1.5726532392076518, 1.5766380165663136, 1.572874354999242], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.4452659574468085, 'nll': 6.927352335503761}, 'chosen_samples': [107378, 50543, 78903, 106619, 79619], 'chosen_samples_score': [1.5055825617567353, 1.5193299943024683, 1.5271796399570474, 1.5450128376282413, 1.5583614027420762], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.44861702127659575, 'nll': 6.779962339604155}, 'chosen_samples': [4444, 13756, 64650, 80430, 531], 'chosen_samples_score': [1.6650573901216945, 1.6775471687207575, 1.7148802857384045, 1.8137301394407772, 1.6807735253276253], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.43829787234042555, 'nll': 6.364445491953099}, 'chosen_samples': [47182, 108249, 54519, 72043, 102521], 'chosen_samples_score': [1.4073235335102727, 1.410495662929828, 1.4341187160316808, 1.4735447165196263, 1.4375419054356349], 'chosen_samples_orignal_score': None})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.44372340425531914, 'nll': 6.168169461108269}, 'chosen_samples': [61599, 92328, 97058, 24240, 37715], 'chosen_samples_score': [1.4958223510210722, 1.5040250326721676, 1.5195869019069423, 1.589133268865913, 1.5900018987478042], 'chosen_samples_orignal_score': None})
