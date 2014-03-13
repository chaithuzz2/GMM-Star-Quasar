GMM-Star-Quasar
========================

Star - quasar classification using Gaussian Mixture Models

I wrote a Gaussian mixtures based model to classify stars and quasars using photometric color measurements provided by Sloan Digital Sky Suvey.

The repository contains three python source files and a data folder in which we have the training data for our method.

classifier.py - checks the accuracy of our GMM classifier 

CLASSIFIER.py - checks the accuracy of scikit-learn's GMM classifier 

GMM.py - Class Implementation of Gaussian Mixture model based classifier 


Requirements
========================

The program is written in Python and also makes use of numpy and scikit-learn modules.

To install numpy:

    sudo apt-get install python-numpy
    
To install scikit-learn instructions can be found here http://scikit-learn.org/stable/install.html


Using the GMM Classifer
========================
    import GMM
    # Initialize the classifier
    classifier = GMM.GMM_Classifier(number_of_components=2, convergence_threshold = 1e-3, number_of_iterations =100)
    # Train the classifier
    classifier.train(xtrain)
    #Calculate the loglikelihood for testing samples
    LogL = classifier.score(xtest)


Output of our program
=========================

The classifier.py program outputs the accuracy of our GMM classifier and the CLASSIFIER.py outputs the accuracy of scikit-learn's GMM classifier.


Remarks
=========================

It works accurately but performance needs to be improved. Any suggestions to do so are extremely welcome :)
