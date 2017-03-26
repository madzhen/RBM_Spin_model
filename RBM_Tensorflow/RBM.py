import tensorflow as tf
import numpy as np
import os
import rbm_kl
import matplotlib.pyplot as plt
import time
import argparse
import import_MC_samples
FLAGS = None

#----------------------------you can change these parameters --------------------------------#


#------------------------------------------------------------------------------------------


def main():
    def GetFileNameAndExt(filename):
     import os
     (filepath,tempfilename) = os.path.split(filename);
     (shotname,extension) = os.path.splitext(tempfilename);
     return shotname,extension


    sample_filename,sample_ext=GetFileNameAndExt(FLAGS.TrainSet_File_name)
    Checkset_FILE="Training_data/check_set_%d.npy"
    CSV_FILE='Training_data/'+sample_filename+'-MC_Dec_sample_%d_spins.csv'
    Result_dir=sample_filename+'-Result_data'




    if not os.path.exists(Result_dir):
            os.mkdir(Result_dir)

    if not os.path.exists('Training_data'):
            os.mkdir('Training_data')   
    print('hello')
    
    print('hello')
    try:
        Traing_set=import_MC_samples.read_data_sets("Training_data/",FLAGS.TrainSet_File_name)
    except Exception as e:
        print("##################################")
        print("\n")
        print('\n-- The traing file not existed, maybe you put it in the wrong folder  ????\n', e)
        print('\n-- please make sure the training_npy file is in the \"Training_data folder\". \n')
        print("\n")
        print("##################################")
        exit()

    H_data=0
    Spin_number = Traing_set.train.Spin_number
    train_set_len=Traing_set.train.num_samples


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
            Dec_sample.append(Return_SpinarrayToDec(Traing_set.train.samples[i]))
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
    rbm_w = tf.placeholder(tf.float64, [Spin_number, FLAGS.hidden_number])
    rbm_a = tf.placeholder(tf.float64, [Spin_number])
    rbm_b = tf.placeholder(tf.float64, [FLAGS.hidden_number])
    train_sigma = tf.placeholder(tf.float64, [None,Spin_number ])
    h0 = sample_prob(tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))


    def sample_prob_h_v(k_step):
        h=h0
        for step in range(k_step):
            v = sample_prob(tf.nn.sigmoid(tf.matmul(h, tf.transpose(rbm_w)) + rbm_a))
            h = sample_prob(tf.nn.sigmoid(tf.matmul(v, rbm_w) + rbm_b))
        return h,v


    h_CDk_k,sigma_CDk_k=sample_prob_h_v(FLAGS.CDk_step)
    w_positive_grad = tf.matmul(tf.transpose(sigma), tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))
    w_negative_grad = tf.matmul(tf.transpose(sigma_CDk_k), tf.nn.sigmoid(tf.matmul(sigma_CDk_k, rbm_w) + rbm_b) )
    update_w = rbm_w + FLAGS.alpha *  (w_positive_grad - w_negative_grad) / tf.to_double(tf.shape(sigma)[0])
    update_vb = rbm_a + FLAGS.alpha * tf.reduce_mean(sigma - sigma_CDk_k, 0)
    update_hb = rbm_b + FLAGS.alpha * tf.reduce_mean(h0 - h_CDk_k, 0)
    h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(sigma, rbm_w) + rbm_b))
    v_sample = sample_prob(tf.nn.sigmoid( tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_a))



    err = sigma - v_sample
    err_sum = tf.reduce_mean(err * err)

    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)



    n_w=np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(Spin_number, FLAGS.hidden_number))
    n_vb = np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(Spin_number))
    n_hb = np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(FLAGS.hidden_number))
    o_w=np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(Spin_number, FLAGS.hidden_number))
    o_vb = np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(Spin_number))
    o_hb = np.random.uniform(-1*FLAGS.initial_random_range,FLAGS.initial_random_range,size=(FLAGS.hidden_number))



    print("The begining transfer error is",sess.run(err_sum, feed_dict={sigma: Traing_set.train.samples, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb}))


    E_sigma_train=tf.squeeze(tf.matmul(sigma,tf.expand_dims(rbm_a,1)))+tf.reduce_sum(tf.log(1+tf.exp(rbm_b+tf.matmul(sigma,rbm_w))),1)
    E_sigma_AllSpinConfiguration=tf.squeeze(tf.matmul(sigma,tf.expand_dims(rbm_a,1)))+tf.reduce_sum(tf.log(1+tf.exp(rbm_b+tf.matmul(sigma,rbm_w))),1)
    tf_ln_Z_sum=tf.log(tf.reduce_sum(tf.exp(E_sigma_AllSpinConfiguration)))



    KL_Y=[]
    KL_X=[]


    for step in range(FLAGS.max_train_Step):
        batch = Traing_set.train.next_batch(FLAGS.batch_size)

        n_w = sess.run(update_w, feed_dict={
                       sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
        n_vb = sess.run(update_vb, feed_dict={
                        sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
        n_hb = sess.run(update_hb, feed_dict={
                        sigma: batch, rbm_w: o_w, rbm_a: o_vb, rbm_b: o_hb})
        o_w = n_w
        o_vb = n_vb
        o_hb = n_hb

        if(step%50==0):
            if(FLAGS.calculate_Transfer_Error==True):
                err_sum_out=sess.run(err_sum, feed_dict={sigma: Traing_set.train.samples, rbm_w: n_w, rbm_a: n_vb, rbm_b: n_hb})
                print('Train Step:'+str(step) +"-----"+'Transfer error(just for reference): '+str(err_sum_out)+"\n")
            else:
                print('Train Step:'+str(step))
            if FLAGS.calculate_KL==True :
                ln_Z_sum=sess.run(tf_ln_Z_sum,feed_dict={sigma:check_sigma_set,rbm_w:o_w,rbm_a:o_vb,rbm_b:o_hb})
                tf_KL_left=(tf.reduce_sum(E_sigma_train)/tf.to_double(tf.shape(sigma)[0]))-ln_Z_sum
                KL_left=sess.run(tf_KL_left,feed_dict={sigma:Traing_set.train.samples,rbm_w:o_w,rbm_a:o_vb,rbm_b:o_hb})
                KL=-1*(KL_left+H_data)
                KL_Y.append(KL)
                KL_X.append(step)
                print("KL is %.6f" % KL)

            if (True==FLAGS.Save_W_a_b_Every_step):
                np.save(Result_dir+'/a_v%d_h%d_s%d.npy' %(Spin_number,FLAGS.hidden_number,step),n_vb)
                np.save(Result_dir+'/b_v%d_h%d_s%d.npy' %(Spin_number,FLAGS.hidden_number,step),n_hb)
                np.save(Result_dir+'/w_v%d_h%d_s%d.npy' %(Spin_number,FLAGS.hidden_number,step),n_w)


    if FLAGS.calculate_KL==True:
        fig = plt.figure(figsize=(31, 16))
        plt.semilogy(KL_X,KL_Y,'o',ms=6,color='blue',label='RBM distribution')
        plt.xlabel('Steps',fontsize=40)
        plt.ylabel('KL divergence (N=%d sites)' %Spin_number,fontsize=30)
        plt.savefig(Result_dir+"/%d_sites_KL_with_cdk %d.png" %(Spin_number,FLAGS.CDk_step) ,dpi=144)
        np.save(Result_dir+"/KL_divergence_%d_sites_with_cdk %d.npy" %(Spin_number,FLAGS.CDk_step),KL)


    #----------------------------------------------save weights matrix-----------------------------------------------------#

    np.save(Result_dir+'/a_v%d_h%d.npy' %(Spin_number,FLAGS.hidden_number),n_vb)
    np.save(Result_dir+'/b_v%d_h%d.npy' %(Spin_number,FLAGS.hidden_number),n_hb)
    np.save(Result_dir+'/w_v%d_h%d.npy' %(Spin_number,FLAGS.hidden_number),n_w)


    print("################")
    print("################")
    print("################")
    print("All Done! Enjoy")
    print("################")
    print("################")
    print("################")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='RBM(Restricted Boltzmann Machines) algorithm on spin model',
            epilog='End of description.')
    parser.add_argument('-a','--alpha', type=float, default=1.0,
                      help='learning rate (default 1.0)')
    parser.add_argument('-cdk','--CDk_step', type=int, default=2,
                      help='Contrastive Divergence step (default 2)')
    parser.add_argument('-irr','--initial_random_range', type=float, default=0.05,
                      help='initial W,a,b with random number in range (-x,x).(default 0.05)')
    parser.add_argument('-bs','--batch_size', type=int, default=3000,
                      help='batch size of training (default 3000)')
    parser.add_argument('-te','--calculate_Transfer_Error', type=bool, default=True,
                      help='calculate Transfer Error (default True)')
    parser.add_argument('-kl','--calculate_KL', type=bool, default=True,
                      help='calculate KL divergence (default True)')
    parser.add_argument('-hn','--hidden_number', type=int, default=12,
                      help='number of hidden nodes (default 12)')
    parser.add_argument('-mts','--max_train_Step', type=int, default=5000,
                      help='total training step (default 5000)')
    parser.add_argument('-ti','--Training interval', type=int, default=50,
                      help='Skip some steps to print the KL and error.(default 50)')
    parser.add_argument('-swe','--Save_W_a_b_Every_step', type=bool, default=True,
                      help='Save W,a,b, in every training loop.(default True)')
    parser.add_argument('-TrainSet_File_name', type=str, default="Ising_demo.npy",
                  help='The file name of training set, sampled from MC.(default Ising_demo.npy)')
    FLAGS = parser.parse_args()
    main()