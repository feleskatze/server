import os, glob

def remove_tmp():
    rmlist = glob.glob('./tmp/*')
    for rmfile in rmlist:
        os.remove(rmfile)
        print(rmfile)

if __name__ == '__main__':
    remove_tmp()
