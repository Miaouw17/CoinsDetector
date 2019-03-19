import glob, os

def rename(directory, pattern):
    i = 1
      
    for filename in os.listdir(directory): 
        dst = pattern + str(i) + ".jpg"
        src = directory + filename 
        dst = directory + dst 
        #print(directory)
        #print(src)  
        #print(dst)

        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1

rename(r'C:/Users/quentin.michel/Documents/hearc/ANNEE-3/Traitement_Image/Projet/CoinsDetector/pieces/train/5c/', r'5c-')