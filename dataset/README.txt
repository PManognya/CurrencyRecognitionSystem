dataset/  = real banknotes, one folder per note, named <CURRENCY CODE>_<DENOMINATION>

  INR_10 INR_20 INR_50 INR_100 INR_200 INR_500 INR_2000     (India)
  TRY_5 TRY_10 TRY_20 TRY_50 TRY_100 TRY_200                (Turkey)
  USD_1 USD_2 USD_5 USD_10 USD_20 USD_50 USD_100            (USA)

negative/ (next to this folder) = images that are NOT banknotes, one sub-folder per kind:
  person face phone laptop book table wall clothes random_objects background hard_negatives

Fill dataset/ automatically with:   python prepare_data.py
Create empty folders:               python make_folders.py
