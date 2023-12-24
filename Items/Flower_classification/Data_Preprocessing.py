import os
import shutil

def data_process():
    i = 0
    for root,dirs,files in os.walk('flower_photos'):
        if i !=0:
            num = len(files)*0.9
            j = 1
            addr = root.split('\\')[1]
            for file in files:

                if j<=num:
                    with open('Dataset/train_label.csv', 'a') as f:
                        shutil.copyfile('./flower_photos/' + addr + '/' + file, './Dataset/train_image/'+addr+str(j)+'.jpg')
                        f.write(addr+str(j)+'.jpg'+','+str(i-1)+'\n')
                else:
                    with open('Dataset/test_label.csv', 'a') as f:
                        shutil.copyfile('./flower_photos/' + addr + '/' + file, './Dataset/test_image/'+addr+str(j)+'.jpg')
                        f.write(addr+str(j)+'.jpg'+','+str(i-1)+'\n')
                j+=1

        i+=1

    print('writing finish')

data_process()
