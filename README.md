# RNN_Stock_Prediction

## Introduction
In January 2020, COVID-19 has started and spread all over the world, and the stock market and local economy have crushed. The Dow Jones and Standard & Poor’s showed that U.S. share values dropped by 20% since mid-March 2020 (Machmuddah, 2020). It took more than half a year to recover the economy and stock shares fall dragged young 20-30s people’s interests. Robinhood, a stock and cryptocurrency trading app, gained 3.1 million users, half of them first-time investors in the first quarter of 2020 (Gao, 2020). Likewise, young people became interested in stocks and bitcoins while taking the risks of losing money as well. In this project, recurrent neural network (RNN), Long Short Time Memory (LSTM), and Gated Recurrent Unit (GRU) models will be used to predict Bitcoins and these models would provide insights to the new investors. In this project, deep learning-driven models are built and compared to find the best models for the prediction.

## Data & Data Preprocessing
The data obtained from Polenix, a cryptocurrency exchange website and the Poloniex package has been uploaded to PyPI. I requested a private API from Polenix to import the current Bitcoin prices. Private API allows customers to read and write operations on the account, but I just imported Bitcoin prices from the websites. The starting date of Bitcoin is 2014 and the ending date is the most current date, which makes the dataset increase daily. The data consisted of open, close, low, and high prices of each day, and close prices were used for the modeling. For the data preprocessing, MinMax scaler, a skit-learn library, is used to normalize the data. In the equation, min represents the feature range parameter, which is the size range of the result (Zhang, 2019).

## References
Z. Machmuddah, St.D Utomo, E. Suhartono, S. Ali, and W. Ghulam, Stock Market Reaction to COVID-19: Evidence in Customer Goods Sector with the Imlication for Open Innovation (2020), Journal of Open Innovation, 6, 99, doi. 10.3390

M. Gao, College Students are Buying Stocks – but Do They Know What They’re Doing? (2020), CNBC, https://www.cnbc.com/2020/08/04/college-students-are-buying-stocks-but-do-they-know-what-theyre-doing.html

T. Zhang, S. Song, L Ma, S. Pan, L. Pan, Research on Gas Concentration Prediction Models Based on LSTM Multidimensional Time Series (2019), Energies, 12(1): 161, doi.10.3390

P T. Yamak, L. Yuigian, and P.K Gadosey, A Comparison Between ARIMA, LSTM, and GRU for Time Series Forecasting 

S. Selvin, Rr. Vinayakumar, E. A. Gopalakrrishnan, and P. Soman, Stock Price Preidctions Using LSTM, RNN, and CNN Sliding Window Model (2017). International Conference on Advances in Computing, Comminucations and Informatics (ICACCI), pp 1643-1647, doi:10.1109

A. Sethia, P. Raut, and D. Sanghiv, Application of LSTM, GRU and ICA for Stock Price Prediction (2018), Smart Innovation, Systems and Technologies, pp 479 – 487, doi:10.1007
