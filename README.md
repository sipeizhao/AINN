# AINN introduction
Source code for (decoupled) Acoustics-Informed Neural Network (AINN) for sound field reconstruction and acoustic beamforming in the following two papers:

Fei Ma, Sipei Zhao and Ian S. Burnett (2024). Sound field reconstruction using a compact acoustics-informed neural network, The Journal of the Acoustical Society of America, 156(3): 2009-2021.
https://pubs.aip.org/asa/jasa/article/156/3/2009/3314199

Sipei Zhao and Fei Ma (2024). A circular microphone array with virtual microphones based on acoustics-informed neural networks, The Journal of the Acoustical Society of America, 156(1): 405-415.
https://pubs.aip.org/asa/jasa/article/156/1/405/3303425/A-circular-microphone-array-with-virtual

The room impulse response (RIR) dataset is available at: https://data.research.uts.edu.au/publication/fad2f4b0c03d11ec91ce05dbccc55a63/

The detailed description of the RIR dataset is available in the following paper:

Sipei Zhao, Qiaoxi Zhu, Eva Cheng and Ian S. Burnett (2022). A room impulse response database for multizone sound field reproduction, The Journal of the Acoustical Society of America, 152(4): 2505-2512.
https://pubs.aip.org/asa/jasa/article-abstract/152/4/2505/2839595/A-room-impulse-response-database-for-multizone?redirectedFrom=fulltext

Sipei Zhao, Qiaoxi Zhu, Eva Cheng and Ian S. Burnett (2024). Erratum: A room impulse response database for multizone sound field reproduction, The Journal of the Acoustical Society of America, 155(3): 2170-2170.

# Source code
The Dataloader.py file reads the time-domain RIRs from the .wav files in the database and convert them to the Room Transfer Functions (RTFs) via Fast Fourier Transform (FFT).
The main.py file reads the frequency domain RTFs to train and test the AINN, which is constructed in the AINN.py file.

