from sys import argv
script, input_file = argv

def print_all(f):
    print f.read()

def rewind(f):
    f.seek(0)
    
    
def print_a_line(line_count,f):
    print line_count, f.readline()
    
current_file = open(input_file)

print "First let's print the whole file:\n"

print_all(current_file)

print "Now let's rewind, kind of like a tape."

rewind(current_file)

print "Let's print three lines:"

current_line = 1
print_a_line(current_line, current_file)

current_line +=1
print_a_line(current_line, current_file)

current_line +=1
print_a_line(current_line, current_file)



#pandas的DataFrame操作
#https://blog.csdn.net/zutsoft/article/details/51483710

#上传github文件
#https://blog.csdn.net/songcy1405/article/details/80484386

#每天一个小算法-线搜索
#https://blog.csdn.net/lvchahp/article/details/39967857


