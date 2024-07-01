def get_filepaths(coin):

  daily_data = None
  hourly_data = None

  if coin == 'Bitcoin':
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\data\Binance_BTCEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\data\Binance_BTCEUR_1h.csv"

  elif coin == "Ethereum":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_ETHEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_ETHEUR_1h.csv"

  elif coin == "Tether":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_EURUSDT_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_EURUSDT_1h.csv"

  elif coin == "Binance Coin":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_BNBEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_BNBEUR_1h.csv"

  elif coin == "Bitcoin Cash":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_BCHEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_BCHEUR_1h.csv"

  elif coin == "Litecoin":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_LTCEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_LTCEUR_1h.csv"

  elif coin == "Internet Computer":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_ICPEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_ICPEUR_1h.csv"

  elif coin == "Polygon":
     daily_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_MATICEUR_d.csv"
     hourly_data = r"C:\Users\JNoot\Documents\University\Bachelor Thesis\New Code\Data\Binance_MATICEUR_1h.csv"

  return daily_data, hourly_data