import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt









#----------------------------you can change these parameters --------------------------------#

Spin_number=6
hidden_number=12
Traing_set="Ising_demo.npy"    # we just need the file name

#------------------------------------------------------------------------------------------




def GetFileNameAndExt(filename):
 import os
 (filepath,tempfilename) = os.path.split(filename);
 (shotname,extension) = os.path.splitext(tempfilename);
 return shotname,extension


sample_filename,sample_ext=GetFileNameAndExt(Traing_set)




Result_dir=sample_filename+'-Result_data'


try:
    check_sigma_set =np.load('Training_data/check_set_%d.npy' %Spin_number)
except Exception as e:
	print('\n-- The check_sigma_set not existed, maybe you put it in the wrong folder  ????\n', e)
	exit()



n_vb=np.load(Result_dir+'/a_v%d_h%d.npy' %(Spin_number,hidden_number))
n_hb=np.load(Result_dir+'/b_v%d_h%d.npy' %(Spin_number,hidden_number))
n_w=np.load(Result_dir+'/w_v%d_h%d.npy' %(Spin_number,hidden_number))



##############################################################-----Plot RBM distribution----###########################################################################
check_set_len=check_sigma_set.shape[0]
P_rbm_numerator_with_all_spin_configuration=np.zeros([check_set_len], np.float32)
tf_P_rbm_numerator_with_all_spin_configuration=tf.placeholder(tf.float32,[check_set_len])
sigma = tf.placeholder(tf.float32, [None,Spin_number ])
rbm_w = tf.placeholder(tf.float32, [Spin_number, hidden_number])
rbm_a = tf.placeholder(tf.float32, [Spin_number])
rbm_b = tf.placeholder(tf.float32, [hidden_number])



sess = tf.InteractiveSession()

E_sigma_AllSpinConfiguration=tf.squeeze(tf.matmul(sigma,tf.expand_dims(rbm_a,1)))+tf.reduce_sum(tf.log(1+tf.exp(rbm_b+tf.matmul(sigma,rbm_w))),1)



tf_P_rbm_numerator=tf.exp(E_sigma_AllSpinConfiguration)
P_rbm_numerator_with_all_spin_configuration=sess.run(tf_P_rbm_numerator,feed_dict={sigma:check_sigma_set,rbm_w:n_w,rbm_a:n_vb,rbm_b:n_hb})
tf_Z_sum=tf.reduce_sum(tf_P_rbm_numerator_with_all_spin_configuration)
Z_sum=sess.run(tf_Z_sum,feed_dict={tf_P_rbm_numerator_with_all_spin_configuration:P_rbm_numerator_with_all_spin_configuration})

P_rbm_distribution=P_rbm_numerator_with_all_spin_configuration/Z_sum


fig = plt.figure(figsize=(31, 16))
x=np.arange(2**Spin_number)
plt.semilogy(x,P_rbm_distribution,'o-',ms=2,lw=1.0,color='blue',label='RBM distribution')
#plt.axis([0,2**Spin_number-1,10**(-7),1])
plt.xlabel('$\sigma$',fontsize=25)
plt.ylabel('P($\sigma $)',fontsize=25)

params = {'legend.fontsize': 20,
          'legend.linewidth': 2}
plt.rcParams.update(params)
plt.title("RBM Learning Boltzmann distribution of model with N=%d sites" %Spin_number,fontsize=30) 
plt.legend()
plt.show()
#	plt.savefig("RBM N=%d sites.png" %Spin_number ,dpi=144)



