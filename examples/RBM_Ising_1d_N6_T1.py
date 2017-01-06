import tensorflow as tf
import numpy as np
import os
import rbm_kl
import matplotlib.pyplot as plt
import time




#----------------------------you can change these parameters --------------------------------#



Traing_set="Ising_demo_1d_N6_T1.npy"
Train_Step=100
CDk_step=2
hidden_number=12
calculate_KL=True

alpha = 1
batchsize = 3000
initial_random_range=0.05  # initial W,a,b with random number in range (-x,x) 
Save_W_a_b_Every_step=False





ministep=20  #calculate KL and Transfer_Error each 20 steps.      The total training step = Train_Step*ministep
calculate_Transfer_Error=True


#------------------------------------------------------------------------------------------




def GetFileNameAndExt(filename):
 import os
 (filepath,tempfilename) = os.path.split(filename);
 (shotname,extension) = os.path.splitext(tempfilename);
 return shotname,extension


sample_filename,sample_ext=GetFileNameAndExt(Traing_set)
Checkset_FILE="Training_data/check_set_%d.npy"
CSV_FILE='Training_data/'+sample_filename+'-MC_Dec_sample_%d_spins.csv'
Result_dir=sample_filename+'-Result_data'





if not os.path.exists(Result_dir):
        os.mkdir(Result_dir)

if not os.path.exists('Training_data'):
        os.mkdir('Training_data')

try:
    trsigma =np.load("Training_data/"+Traing_set)
except Exception as e:
    print("##################################")
    print("\n")
    print('\n-- The traing file not existed, maybe you put it in the wrong folder  ????\n', e)
    print('\n-- please make sure the training_npy file is in the \"Training_data folder\". \n')
    print("\n")
    print("##################################")
    exit()

H_data=0
Spin_number = trsigma.shape[1]
train_set_len=trsigma.shape[0]


#------------------------------Generate check_set.npy and MC_Dec_sample_.csv-------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#




def Return_SpinConfiguration_array(Decimal_num, N_number_of_spin_sites):
    spin_array = np.asarray(
        [1 if int(x) else 0 for x in
         "".join([str((Decimal_num >> y) & 1) for y in range(N_number_of_spin_sites - 1, -1, -1)])])
    return spin_array
def Return_SpinarrayToDec(spin_array):
    result = 0
    for i in range(len(spin_array)):
        result = result * 2 + (1 if spin_array[i] == 1 else 0)
    return result
def Generate_checkset(N_number_of_spin_sites):
    check_set=[]
    for i in range(2**N_number_of_spin_sites):
        check_set.append(Return_SpinConfiguration_array(i,N_number_of_spin_sites))
    out=np.asarray(check_set)   
    np.save(Checkset_FILE %N_number_of_spin_sites,out)
    check_set=[]


if (os.path.exists(Checkset_FILE %Spin_number)):
    print("------   check_set.npy already exist\n")
    check_sigma_set =np.load(Checkset_FILE %Spin_number)
else:
    print("This is the 1st time you run this code,we need few minutes to generate check_set_N.npy,please wait\n")
    print('\n')
    print("--------- Generating check_set,please wait........\n")
    Generate_checkset(Spin_number)
    print("checkset.npy saved successfully\n")
    check_sigma_set =np.load(Checkset_FILE %Spin_number)



if (os.path.exists(CSV_FILE %Spin_number)):
    print("------   MC_Dec_sample.csv already exist\n")
    print("\n")
    H_data=rbm_kl.calculate_H_data(CSV_FILE %Spin_number)  
    print("############################################\n")
    print("------      The H_data is: ",H_data)
    print("############################################\n")
    print("\n")
else:
    print("This is the 1st time you run this code,we need few minutes to generate MC_Dec_sample.csv,please wait\n")
    print('\n')
    print("--------- Generating csv,please wait........\n")
    Dec_sample=[]
    for i in range(train_set_len):
        Dec_sample.append(Return_SpinarrayToDec(trsigma[i]))
    Dec_sample=np.asarray(Dec_sample)
    outfile=(CSV_FILE %Spin_number)
    with open(outfile, 'wb+') as fout:
    	np.savetxt(fout, Dec_sample,fmt='%d') 
    	fout.seek(-1, 2)
    	fout.truncate()
    Dec_sample=[]
    print("MC_Dec_sample_%d_spins.csv saved successfully\n" %Spin_number)
    H_data=rbm_kl.calculate_H_data(CSV_FILE %Spin_number)
    print("\n") 
    print("############################################\n")
    print("------      The H_data is: ",H_data)
    print("############################################\n")
    print("\n")
#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#













print("####  Please make sure all initial file are correct,Now we start. ####\n")
time.sleep(2)

check_set_len=check_sigma_set.shape[0]


def sample_prob(probs):
    return tf.nn.relu( tf.sign( probs - tf.random_uniform(tf.shape(probs),dtype=tf.float64)))



sigma = tf.placeholder(tf.float64, [None,Spin_number ])
rbm_w = tf.placeholder(tf.float64, [Spin_number, hidden_number])
rbm_a = tf.placeholder(tf.float64, [Spin_number])
rbm_b = tf.placeholder(tf.float64, [hidden_number])
train_sigma = tf.placeholder(tf.float64, [None,Spin_number ])



h0 = sample_prob(tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))
def sample_prob_h_v(k_step):
    h=h0
    for step in range(k_step):
        v = sample_prob(tf.nn.sigmoid(tf.matmul(h, tf.transpose(rbm_w)) + rbm_a))
        h = sample_prob(tf.nn.sigmoid(tf.matmul(v, rbm_w) + rbm_b))
    return h,v


h_CDk_k,sigma_CDk_k=sample_prob_h_v(CDk_step)
w_positive_grad = tf.matmul(tf.transpose(sigma), tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))
w_negative_grad = tf.matmul(tf.transpose(sigma_CDk_k), tf.nn.sigmoid(tf.matmul(sigma_CDk_k, rbm_w) + rbm_b) )
update_w = rbm_w + alpha *  (w_positive_grad - w_negative_grad) / tf.to_double(tf.shape(sigma)[0])
update_vb = rbm_a + alpha * tf.reduce_mean(sigma - sigma_CDk_k, 0)
update_hb = rbm_b + alpha * tf.reduce_mean(h0 - h_CDk_k, 0)
h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))
v_sample = sample_prob(tf.nn.sigmoid( tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_a))



err = sigma - v_sample
err_sum = tf.reduce_mean(err * err)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)




n_w=np.random.uniform(-1*initial_random_range,initial_random_range,size=(Spin_number, hidden_number))
n_vb = np.random.uniform(-1*initial_random_range,initial_random_range,size=(Spin_number))
n_hb = np.random.uniform(-1*initial_random_range,initial_random_range,size=(hidden_number))
o_w=np.random.uniform(-1*initial_random_range,initial_random_range,size=(Spin_number, hidden_number))
o_vb = np.random.uniform(-1*initial_random_range,initial_random_range,size=(Spin_number))
o_hb = np.random.uniform(-1*initial_random_range,initial_random_range,size=(hidden_number))





print("The begining transfer error is",sess.run(err_sum, feed_dict={sigma: trsigma, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb}))


E_sigma_train=tf.squeeze(tf.matmul(sigma,tf.expand_dims(rbm_a,1)))+tf.reduce_sum(tf.log(1+tf.exp(rbm_b+tf.matmul(sigma,rbm_w))),1)
E_sigma_AllSpinConfiguration=tf.squeeze(tf.matmul(sigma,tf.expand_dims(rbm_a,1)))+tf.reduce_sum(tf.log(1+tf.exp(rbm_b+tf.matmul(sigma,rbm_w))),1)
tf_ln_Z_sum=tf.log(tf.reduce_sum(tf.exp(E_sigma_AllSpinConfiguration)))
Shuffle_tf_trsigma=tf.random_shuffle(train_sigma)






KL_Y=[]
KL_X=[]
training_step=0
batch_step=0





trsigma=sess.run(Shuffle_tf_trsigma,feed_dict={train_sigma:trsigma})


for step_outloop in range(Train_Step*ministep):
    batch_next = trsigma[batch_step:batch_step+batchsize,]
    batch_step=batch_step+batchsize
    if(batch_step>(trsigma.shape[0]-1)):
        trsigma=sess.run(Shuffle_tf_trsigma,feed_dict={train_sigma:trsigma})
        batch_step=0


    batch=batch_next
    n_w = sess.run(update_w, feed_dict={
                   sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
    n_vb = sess.run(update_vb, feed_dict={
                    sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
    n_hb = sess.run(update_hb, feed_dict={
                    sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb


    if step_outloop % ministep == 0:
        training_step=training_step+1
        if(calculate_Transfer_Error==True):
            err_sum_out=sess.run(err_sum, feed_dict={sigma: trsigma, rbm_w: n_w, rbm_a: n_vb, rbm_b: n_hb})
            print('Train Step:'+str(training_step) +"-----"+'Transfer error(just for reference): '+str(err_sum_out)+"\n")
        else:
            print('Train Step:'+str(training_step))
        if calculate_KL==True :
            ln_Z_sum=sess.run(tf_ln_Z_sum,feed_dict={sigma:check_sigma_set,rbm_w:o_w,rbm_a:o_vb,rbm_b:o_hb})
            tf_KL_left=(tf.reduce_sum(E_sigma_train)/tf.to_double(tf.shape(sigma)[0]))-ln_Z_sum
            KL_left=sess.run(tf_KL_left,feed_dict={sigma:trsigma,rbm_w:o_w,rbm_a:o_vb,rbm_b:o_hb})
            KL=-1*(KL_left+H_data)
            KL_Y.append(KL)
            KL_X.append(training_step)
            print("KL is %.6f" % KL)

        if (True==Save_W_a_b_Every_step):
            np.save(Result_dir+'/a_v%d_h%d_s%d.npy' %(Spin_number,hidden_number,training_step),n_vb)
            np.save(Result_dir+'/b_v%d_h%d_s%d.npy' %(Spin_number,hidden_number,training_step),n_hb)
            np.save(Result_dir+'/w_v%d_h%d_s%d.npy' %(Spin_number,hidden_number,training_step),n_w)


if calculate_KL==True:
    fig = plt.figure(figsize=(31, 16))
    plt.semilogy(KL_X,KL_Y,'o',ms=6,color='blue',label='RBM distribution')
    plt.xlabel('Steps',fontsize=40)
    plt.ylabel('KL divergence (N=%d sites)' %Spin_number,fontsize=30)
    plt.savefig(Result_dir+"/%d_sites_KL_with_cdk %d.png" %(Spin_number,CDk_step) ,dpi=144)



#----------------------------------------------save weights matrix-----------------------------------------------------#

np.save(Result_dir+'/a_v%d_h%d.npy' %(Spin_number,hidden_number),n_vb)
np.save(Result_dir+'/b_v%d_h%d.npy' %(Spin_number,hidden_number),n_hb)
np.save(Result_dir+'/w_v%d_h%d.npy' %(Spin_number,hidden_number),n_w)


print("################")
print("################")
print("################")
print("All Done! Enjoy")
print("################")
print("################")
print("################")
