# Tracking Influenza A Virus Infection in the Lung from Hematological Data with Machine Learning
Suneet Singh Jhutty<sup>1,2,#</sup>, Julia D. Boehme<sup>3,4,#</sup>, Andreas Jeron<sup>3,4</sup>, Julia Volckmar<sup>3,4</sup>, Kristin Schultz<sup>3,5</sup>, Jens Schreiber<sup>6</sup>, Klaus Schughart<sup>5,7,8</sup>, Kai Zhou<sup>1</sup>, Jan Steinheimer<sup>1</sup>, Horst Stöcker<sup>1,9,10</sup>, Sabine Stegemann-Koniszewski<sup>6</sup>, Dunja Bruder<sup>3,4*</sup>, Esteban A. Hernandez-Vargas<sup>1,11*</sup>
  
    1) Frankfurt Institute for Advanced Studies, 60438, Frankfurt am Main, Germany
    2) Faculty of Biological Sciences, Goethe University, 60438 Frankfurt am Main, Germany
    3) Immune Regulation Group, Helmholtz Centre for Infection Research, 38124 Braunschweig, Germany
    4) Infection Immunology Group, Institute of Medical Microbiology, Infection Control and Prevention, Health Campus Immunology, Infectiology and Inflammation,  Otto-von-Guericke-University Magdeburg, 39120 Magdeburg, Germany
    5) Infection Genetics Group, Helmholtz Centre for Infection Research, 38124 Braunschweig, Germany
    6) Department of Pneumology, Health Campus Immunology, Infectiology and Inflammation, Otto-von-Guericke University Magdeburg, 39120 Magdeburg, Germany
    7) Department of Microbiology, Immunology, and Biochemistry, University of Tennessee Health Science Center, Memphis, TN 38163, United States
    8) University of Veterinary Medicine Hannover, 30559 Hannover, Germany
    9) Institut für Theoretische Physik, Goethe Universität Frankfurt, 60438 Frankfurt am Main, Germany
    10) GSI Helmholtzzentrum für Schwerionenforschung GmbH, 64291 Darmstadt, Germany
    11) Instituto de Matemáticas, Universidad Nacional Autónoma de México, 76230, Juriquilla, México

    Contributions
    # These authors contributed equally: Suneet Singh Jhutty, Julia D. Boehme
    *  These authors jointly supervised this work: Dunja Bruder, Esteban A. Hernandez-Vargas

# Abstract
The tracking of viral burden and host responses with minimal invasive methods during respiratory infections is central for monitoring disease development and guiding treatment decisions. Utilizing a standardized murine model of respiratory influenza A virus (IAV) infection, we developed and tested different supervised machine learning models to predict viral burden and immune response markers, i.e. cytokines and leukocytes in the lung, from hematological data. We performed independent in vivo infection experiments to acquire extensive data for training and testing purposes of the models. We show here that lung viral load, neutrophils, cytokines like IFN-γ and IL-6, and other lung infection markers can be predicted from hematological data. Furthermore, feature analysis of the models shows that granulocytes and platelets play a crucial role in prediction and are highly involved in the immune response against IAV. The proposed in silico tools pave the path towards improved tracking and monitoring of influenza infections and possibly other respiratory diseases based on minimal-invasively obtained hematological parameters.

# Code
This repository contains the supplementalary simulation code for above paper.
The respository is structured the following way:
- Code: contains the python files to create results & figures
- Data: contains data created produced by us
- Plots: contains figures from the paper and supplemental material
- Results: contains text files with the results (selected metrices) from the execution of the python scripts 

To use the python files, please follow the instructions on top of the file (if any) or simply execute it.

The python code uses the version 3.7.7 and the following libraries:
- Tensorflow: 2.4.1
- Sklearn:    0.24.2
- Numpy:      1.18.5
- Pandas:     1.2.4
- Matplotlib: 3.3.4
- Seaborn:    0.11.1
