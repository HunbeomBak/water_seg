import os

class TextWriter:
    def __init__(self, txt_path):
        
        self.txt_path = txt_path
        
        if not os.path.exists(txt_path):
            self.mk_file()
        
    def mk_file(self):
        f = open(self.txt_path, 'w')
        f.close()
        
    def add_line(self, txt):
        f = open(self.txt_path, 'a')
        f.write(txt)
        f.close()