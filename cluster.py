#+++
# this code runs under python 2.7.2 
# because numpy and matplotlib could not be installed with higher versions of  python
#
# NOTE: under 3.2.2 print requires parenthesis
# print ("hello 3.2.2")
# versus
# print "hello 2.7x" 
#++-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#+++
# the file "Iris.txt.csv" contains 6 fields, as follows
# sepal_length, sepal_width, petal_lenght, petal_width, class, row_number
# the first row contains the names of the fields

# the file "Iris.txt" contains 6 fields, as follows
# sepal_length, sepal_width, petal_lenght, petal_width, class, row_number
# The file does NOT contain the names of the fields

# the file UCI_ML_Iris_Data.txt contains the first 5 fields
# no column with record numbers and no row with titles.
# BUT, there is something wrong with that file. 
# I took the file directly from the source, copied it into a googledoc document, then export it
# interestingly when Iris.txt and UCI_ML_Iris.txt are both open under emacs, the UCI* file
# shows the attribute UUU(DOS) whereas the other file shows UU
# in Emacs:
#  u      = utf-8
#  U      = utf-16
#  (DOS)  = Windows NL convension
#  (Unix) = Unix NL convension 
#
#

#++-

def read_data():
  f=open("Iris.txt", 'r')
  lines=[line.strip() for line in f.readlines()]
  f.close()
  lines=[line.split(",") for line in lines if line]
  # only for Iris.txt.csv
  # row_names = lines[0] 
  class1=np.array([line[:4] for line in lines if line[-2]=="Iris-setosa"], dtype=np.float)
  class2=np.array([line[:4] for line in lines if line[-2]!="Iris-setosa"], dtype=np.float)
  return class1, class2
 

def read_data1():
  #f=open("UCI_ML_Iris_Data.txt", 'r')
  f=open("Iris.txt", 'r')
  lines=[line.strip() for line in f.readlines()]
  f.close()
  lines=[line.split(",") for line in lines if line]
  #only for file 'UCI_ML_Iris_Data.txt'
  class1=np.array([line[:4] for line in lines if line[4]=="Iris-setosa"], dtype=np.float)
  class2=np.array([line[:4] for line in lines if line[4]!="Iris-setosa"], dtype=np.float)
  return class1, class2
 

def main():
  class1, class2=read_data1()
  mean1=np.mean(class1, axis=0)
  mean2=np.mean(class2, axis=0)
  #calculate variance within class
  Sw=np.dot((class1-mean1).T, (class1-mean1))+np.dot((class2-mean2).T, (class2-mean2))
  #calculate weights which maximize linear separation
  w=np.dot(np.linalg.inv(Sw), (mean2-mean1))
  # python 3.3.2
  # print ("vector of max weights", w)
  # python 2.7.2
  print "vector of max weights", w 
  #projection of classes on 1D space
  plt.plot(np.dot(class1, w), [0]*class1.shape[0], "bo", label="Iris-setosa")
  plt.plot(np.dot(class2, w), [0]*class2.shape[0], "go", label="Iris-versicolor and Iris-virginica")
  plt.legend()
  plt.show()
   

main()


