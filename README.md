# SOM_OPV

This repo provides code and notebooks for sompy project. You can find more detailed information from this publication.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to download anaconda and install python packages in the requirement.txt file using command below:

```
pip install -r requirement.txt
```

### Installing

tfprop_sompy is the fundamental framework we base on. In order to use that, simply follow instruction in the package

## Content

In the `Data` folder, the `compdata_classificatiion_des.csv` is the data file with molecular descriptors appened to it (More than 1800 descriptors are appended, and you can do feature selection based on them). The original data comes from Professor Saeki's paper: [Computer-Aided Screening of Conjugated Polymers for Organic Solar Cell: Classification by Random Forest](https://pubs-acs-org.offcampus.lib.washington.edu/doi/abs/10.1021/acs.jpclett.8b00635). You can find the original data there. 

Molecular descriptors are calculated using a third-party library called `Mordred Descriptor`.

Notebook can be run after installing Python, rdkit (I use Anaconda), mordred and some other libraries. 

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Jingtian Zhang** (https://github.com/Zhangjt9317)
* **Luna Huang** (https://mse.washington.edu/people/faculty/adjunct)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* University of Washington Chem E, MSE, Tohoku University MSE, Grad School of Engineering, etc. 
