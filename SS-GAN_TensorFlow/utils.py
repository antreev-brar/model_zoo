def load_dataset():
  (x_train,y_train),(x_test,y_test) = load_data()
  
  x_train = expand_dims(x_train,axis =-1)
  x_train = (x_train.astype('float32')-127.5)/127.5
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)
  #print(x_train[0])
  return [x_train , y_train]
#load_dataset()

def make_supervised_train_dataset(dataset , num_classes =10 ,num_samples =100):
  X,Y = dataset
  x_ = list()
  y_ = list()
  samples_per_class = int(num_samples / num_classes)
  for k in range(num_classes):
    x_class = X[Y == k]
    print(len(X))
    index = randint(0,len(x_class),samples_per_class)
    print(index)
    [x_.append(x_class[p]) for p in index]
    [y_.append(k) for p in index]
  print(len(x_))
  print(len(y_))
  return asarray(x_) , asarray(y_)
#make_supervised_train_dataset(load_dataset())

def gen_real_samples(dataset , num_samples):
  images , labels = dataset 
  index = randint(0, images.shape[0], num_samples)
  img_sample , label_sample = images[index] , labels[index]
  y = ones((num_samples , 1))
  return [img_sample , label_sample] , y

def gen_fake_samples(generator , latent_dim , num_samples):
  z = generate_latent_points(num_samples,latent_dim)
  fake_samples = generator.predict(z)
  y = zeros((num_samples , 1))
  return fake_samples , y

def generate_latent_points(num_samples , latent_dim):
  latent_input = randn(num_samples * latent_dim)
  latent_input = latent_input.reshape(num_samples , latent_dim)
  #print(latent_input)
  return latent_input
#generate_latent_points(5,5)

def summarize_performance(step, g_model,s_model, latent_dim, dataset, n_samples=100):
    # prepare fake examples
    X, _ = gen_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
      # define subplot
      pyplot.subplot(10, 10, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    X, y = dataset
    _, acc = s_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    
    # save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = 'c_model_%04d.h5' % (step+1)
    s_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
    plot_test_acc.append(acc)

def one_example():
  img , _= gen_fake_samples(g_model , 100,1)
  ii = tf.convert_to_tensor(
    img, dtype=None, dtype_hint=None, name=None
  )
  plt.figure(figsize=(2,2))
  plt.title(np.argmax(s_model.predict(ii, batch_size =None ,callbacks=None , steps =1)))
  plt.grid = False 
  plt.xticks([])
  plt.yticks([])
  img = np.reshape(img ,(28,28))
  plt.imshow(img)

def graph_plot():
  number_steps =epochs_*600
  plt.figure(figsize=(10,10))
  plt.plot(np.arange(0,number_steps),plot_sup_loss)
  plt.plot(np.arange(0,number_steps),plot_unsup_loss)
  plt.plot(np.arange(0,number_steps),plot_gan_loss)
  plt.title('loss_function')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['sup_loss', 'unsup_loss','gan_loss'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.figure(figsize=(10,10))
  plt.plot(np.arange(0,number_steps),plot_sup_acc)
  plt.title('acc_function')
  plt.ylabel('acc')
  plt.xlabel('epoch')
  plt.legend(['train acc_sup'], loc='upper left')
  plt.show()
  # summarize history for train_acc
  plt.figure(figsize=(10,10))
  plt.plot(np.arange(0,number_steps/600),plot_test_acc)
  plt.title('acc_function')
  plt.ylabel('test_acc')
  plt.xlabel('epoch')
  plt.legend(['test_acc'], loc='upper left')
  plt.show()
  # summarize history for test_acc