import numpy as np
import sys
import math

"""Gaussian mixture model(GMM) is a mixture of several Gaussian distributions and can therefore represent different subclasses inside one class
The probability density function is defined as a weighted sum of gaussians. It can be used in classification by fitting a mixture of a  gaussians to a class and maximizing the likelihood of the samples. For more information Check http://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model """



class GMM_Classifier(object):

	"""Initalizing the model with number of components , convergence threshold and number of iterations for the EM algorithm . A diagonal Covariance is 	assumed"""

	def __init__(self, number_of_components = 2, convergence_threshold = 1e-3, number_of_iterations = 100):
		self.number_of_components = number_of_components
		self.convergence_threshold = convergence_threshold
		self.number_of_iterations = number_of_iterations
		self.weights = np.ones(self.number_of_components) / self.number_of_components
		self.is_converged = False		
		self.means = None
		self.variances = None
		self.number_of_attributes = None
		self.number_of_training_samples = None
		self.number_of_testing_samples = None
		self.gamma = None
		self.effective_number_of_samples = None

	"""The following method fits the data to the mixture model by using what is known as Expectation Maximization Technique. Basically what it does is
	It estimates the means and variances by maximizing the likelihood of samples fitting under our model using an iterative process and convergence
	Threshold   """		


	def train(self,xtrain):
		self.number_of_training_samples = len(xtrain)
		self.number_of_attributes = len(xtrain[0])
		self.means = np.empty((self.number_of_components, self.number_of_attributes), float)
		self.variances = np.empty((self.number_of_components, self.number_of_attributes), float)
		self.gamma = np.empty((self.number_of_training_samples, self.number_of_components), float)
		self.effective_number_of_samples = np.zeros((self.number_of_components), float)
		Initial_mean = np.empty((self.number_of_attributes), float)
		Initial_variance = np.empty((self.number_of_attributes), float)
		temp_array = None
		for a in range(0, self.number_of_attributes):
			temp_array = [xtrain[b][a] for b in range(0, self.number_of_training_samples)]
			Initial_mean[a] = np.mean(temp_array)   
		 	Initial_variance[a] = np.std(temp_array)
		for d in range(0, self.number_of_components):
			self.means[d] = [x + np.random.random()-0.5 for x in Initial_mean]
			self.variances[d] =  [x + np.random.random() for x in Initial_variance]
		iterations = 0
		while not self.is_converged:
			logL = self.logLikelihood(xtrain, self.weights, self.means, self.variances) 
			for j in range(0, self.number_of_training_samples):
				for k in range(0, self.number_of_components):
					numerator = self.weights[k]*self.multivariate_PDF(xtrain[j], self.means[k], self.variances[k])
					denominator = self.full_mixture_PDF(xtrain[j], self.weights, self.means, self.variances) 								
		 			self.gamma[j][k] = numerator / denominator
			for l in range(0, self.number_of_components):
		 		temp_sum_N = 0.0
		 		for m in range(0, self.number_of_training_samples):		
					temp_sum_N += self.gamma[m][l]
				self.effective_number_of_samples[l] = temp_sum_N
			for n in range(0, self.number_of_components):
				for q in range(0, self.number_of_attributes):
					temp_sum_mu = 0.0
					for p in range(0, self.number_of_training_samples):	
						temp_sum_mu+= self.gamma[p][n]*xtrain[p][q]
					temp = temp_sum_mu / self.effective_number_of_samples[n]
					self.means[n][q] = temp
			for s in range(0, self.number_of_components):
				for t in range(0, self.number_of_attributes):
					temp_sigmasq = 0.0
					for u in range(0, self.number_of_training_samples):
						temp_sigmasq+= self.gamma[u][s]*((xtrain[u][t] - self.means[s][t])**2)
					self.variances[s][t] = temp_sigmasq / self.effective_number_of_samples[n]
			for v in range(0, self.number_of_components):
				self.weights[v] = self.effective_number_of_samples[v] / self.number_of_training_samples
			temp_logL = self.logLikelihood(xtrain, self.weights, self.means, self.variances)
			if( temp_logL - logL < self.convergence_threshold):
				self.is_converged = True
			if( temp_logL - logL >= self.convergence_threshold):
				logL = temp_logL
			iterations +=1
			if( iterations == self.number_of_iterations):
				self.is_converged = True
			
			 
	""" Computes the log likelihood o the testing sample and outputs it. It is used in class labelling in the caller function """
	def score(self, xtest):
		self.number_of_testing_samples = len(xtest)
		output_score = np.empty((self.number_of_testing_samples), float)
		for z in range(0, self.number_of_testing_samples):
			output_score[z] = math.log(self.full_mixture_PDF(xtest[z], self.weights, self.means, self.variances))
		return output_score


	""" A helper function to calculate log likelihood """


	def logLikelihood(self, xtrain, weights, means, variances):
		log_sum = 0.0
		for w in range(0, self.number_of_training_samples):
			log_sum += math.log(self.full_mixture_PDF(xtrain[w], weights, means, variances))
		return log_sum

	"""Calculates the total pdf of a sample in the gaussian mixture model"""

	def full_mixture_PDF(self, x, full_weights, full_means, full_variances): 
		answer = 0
		for c in range(0, self.number_of_components):
			 answer+= full_weights[c]*self.multivariate_PDF(x, full_means[c], full_variances[c])
		return answer	 	


	""" calculates per component multivariate pdf . Assumes diagonal covariance , which make calculation easier like this. Multivariate pdf with
	Diagonal covariance is nothing but the product of univariate distributions """



	def multivariate_PDF(self, x, means, variances):
		result = np.zeros((self.number_of_attributes), float)
		for i in range(0, self.number_of_attributes):
			result[i] = self.calculatePDF(x[i], means[i], variances[i])
			if(result[i]==0):
				print str(x[i])+" "+ str(means[i])+" " + str(variances[i]) 
		final_result = np.product(result)
		return final_result


	""" A univariate PDF calculator. Smoothed so that it doesnt return 0.0 because of lowest value precision problem"""

	def calculatePDF(self, sample, mean, variance):
		Pi = 3.14159
		Denominator = (2*Pi*variance)**.5
		power = -(float(sample)-float(mean))**2/(2*variance) 
		if(power <= -500.0):
			power = -400.0
		Numerator = math.exp(power)
		return Numerator/Denominator		 		  
	 		
			 

