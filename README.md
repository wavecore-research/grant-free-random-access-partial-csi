# Grant-Free Random Access in Massive MIMO for IoT Nodes with partial CSI
Simulation code

## How to run

1. Install Nvidia drivers for the GPU (see cupy for more information)
2. install conda (miniconda/anaconda)
3. open anaconda prompt and navigate to the project folder
4. create conda environment: `conda env create --file environment.yaml`
5. activate the conda environment: `conda activate grant-free-random-access-partial-csi`
6. run the simulation: `python ./main_v3.py`

## How to cite

```latex
@INPROCEEDINGS{Call2303:Grant,
AUTHOR="Gilles Callebaut and François Rottenberg and Liesbet {Van der Perre} and
Erik G. Larsson",
TITLE="{Grant-Free} Random Access of {IoT} devices in Massive {MIMO} with Partial
{CSI}",
BOOKTITLE="2023 IEEE Wireless Communications and Networking Conference (WCNC) (IEEE
WCNC 2023)",
ADDRESS="Glasgow, United Kingdom (Great Britain)",
DAYS="26",
MONTH=mar,
YEAR=2023,
KEYWORDS="activity detection; grant-free; massive MIMO; maximum likelihood; random
access",
ABSTRACT="The number of wireless devices is drastically increasing, resulting in many
devices contending for radio resources. In this work, we present an
algorithm to detect active devices for unsourced random access, i.e., the
devices are uncoordinated. The devices use a unique, but non-orthogonal
preamble, known to the network, prior to sending the payload data. They do
not employ any carrier sensing technique and blindly transmit the preamble
and data. To detect the active users, we exploit partial channel state
information (CSI), which could have been obtained through a previous
channel estimate. For static devices, e.g., Internet of things nodes, it is
shown that CSI is less time-variant than assumed in many theoretical works.
The presented iterative algorithm uses a maximum likelihood approach to
estimate both the activity and a potential phase offset of each known
device. The convergence of the proposed algorithm is evaluated. The
performance in terms of probability of miss detection and false alarm is
assessed for different qualities of partial CSI and different
signal-to-noise ratio."
}
```
 
