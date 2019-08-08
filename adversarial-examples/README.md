# Attack Samples

Adversarial examples produces by our method both in the physical and 
the digital environment are hosted on Box and can be downloaded from 
[https://tinyurl.com/AGNsAdvExamples](https://tinyurl.com/AGNsAdvExamples). Note that the data reflects a specific experimental 
context. Evaluating the data in a different experimental context is 
unlikely to accurately reflect the efficacy of the methods used to 
generate the data.

Once you download and decompress the linked file, you'll find two 
directories: `physical/` and `digital/`. Under the directory 
`physical/` you can find the samples of physically realized attacks 
against the VGG143 and VGG10 (see our 
[paper](https://users.ece.cmu.edu/~mahmoods/publications/tops19-adv-ml.pdf)). The directory `digital/` contains samples of attacks in the 
digital environment against the same neural networks.

The names of dodging samples are formatted as:
`<NN>-<ATTACKER_IDX>-<COUNTER>.png`

The names of impersonation samples are formatted as:
`<NN>-<ATTACKER_IDX>-<TARGET_IDX>-<COUNTER>.png`
