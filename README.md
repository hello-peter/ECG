**Title : Prediction of ECG Abnormality Based on Recursive Neural Network**  

This project is based on the mxnet framework  
>Directory:  
    ---ECG5000  
    ---mode  
    ---train_vae.py  
    ---train_lstm.py   
    ---train_vae_lstm.py  
    ---README.md  

ECG5000:  
> 1.A folder where data is stored  
>2.Data is introduced  :
>a 20-hour long ECG recorded from a 48- year-old male with severe congestive heart failure. This record has 17,998,834 data points containing 92,584 heartbeats.
The data description address is `[data description][1]`

mode:  
>---lstm.py  
>---vae.py  
>---vae_lstm.py  



Train mode function file:
>train_lstm.py contains functions to train LSTM mode  
>train_vae.py contains functions to train VAE mode  
>train_vae_lstm contains functions to train VAE_LSTM mode  

introduce vae_lstm:
>VAE_LSTM is to input the hidden variable Z generated by the VAE coding layer after training into the modified LSTM model to mine the characteristics of the whole data and time series

Test accuracy:
>LSTM : 95.2953%  
>VAE_LSTM : 98.4985%   

  [1]: http://timeseriesclassification.com/description.php?Dataset=ECG5000# ECG

