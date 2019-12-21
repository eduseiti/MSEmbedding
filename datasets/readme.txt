Datasets binary file description
--------------------------------

"validationSet" is the "test_mixedSpectra_v0.1_small.pkl" file loaded.



['spectra', 'multipleScansSequences', 'singleScanSequences', 'normalizationParameters', 'spectraCount']


'spectra': {<petide-sequence> : <list-of-spectra>}

	<list-of-spectra> = 	list of all spectra mapped to the given <peptide-sequence>.
							Each list element is a dictionary in the following format:

							{
								'nzero_peaks' : <torch.Size([<number-of-pairs>, 2)>, 
								'pepmass' : <list-of-precursor-mass-candidates>, 
								'charge' : <string>
							}


	Example:

		validationSet['spectra']['HGGYKPTDK']

			[{'nzero_peaks': tensor([[-1.1778, -0.0423],
			          [-1.1778, -0.0410],
			          [-1.1778, -0.0402],
			          ...,
			          [ 1.4164, -0.0418],
			          [ 1.4164, -0.0419],
			          [ 1.4164, -0.0421]]), 'pepmass': [501.753875732422], 'charge': '2+'},
			 {'nzero_peaks': tensor([[-1.1778, -0.0422],
			          [-1.1778, -0.0393],
			          [-1.1778, -0.0379],
			          ...,
			          [ 1.3274, -0.0386],
			          [ 1.3274, -0.0393],
			          [ 1.3275, -0.0405]]), 'pepmass': [334.838684082031], 'charge': '3+'}]




'multipleScansSequences': list of all peptide sequences which have more than 1 spectrum mapped to it, i.e., which has more than one 						  element in the 'spectra' key.


	Example:

		validationSet['multipleScansSequences']		

			['HGGYKPTDK',
			 'KLSSAMSAAK',
			 'HHVLHDQNVDKR',
			 'PFGNTHNK',
			 'VLPAHDASK',
			 'RGSNTTSHLHQAVAK',
			 'HGVYNPNK',
			 'HQGVMVGMGQK',
			 'HKTDLNHENLK',
			 'KKGHHEAEIKPLAQSHATK',
			 'KKGHHEAELKPLAQSHATK',
			 'HLKDEMAR',
			 ...]

		len(validationSet['multipleScansSequences'])

			1710




'singleScanSequences': list of all peptide sequences which have a single spectrum mapped to it.

	Example:

		validationSet['singleScanSequences']

			['HLTSNSPR',
			 'HSPSPVR',
			 'HSGPSSYK',
			 'AHAHLDTGR',
			 'GHLIHGR',
			 'VLGKPR',
			 'EHQRPTLR',
			 'VLSANR',
			 'LHTVQPK',
			 'VTGHPKPIVK',
			 'GHHEAEIKPLAQSHATKHK',
			 'GHHEAELKPLAQSHATKHK'
			 ...]

		len(validationSet['singleScanSequences'])

			4122


'normalizationParameters': dictionary with the parameters used to normalize the dataset. These parameters are the z-norm from the
						   training dataset:

	Example:

		validationSet['normalizationParameters']

			{
				'mz_mean': tensor(494.6371),
				'mz_std': 326.5139497402829,
				'intensity_mean': tensor(3463.5706),
				'intensity_std': 81383.90838488896
			}

'spectraCount': the total number of spectra in this dataset, considering all the recognized and the unrecognized sequences.

	9175